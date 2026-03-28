// static/script.js
// Handles: fetch to /predict, charts, LIME highlighting, history, metrics table.

let barChart = null, pieChart = null, trendChart = null;
const trendLabels = [], trendValues = [];
const history = [];

// helper: safe HTML escape
function escapeHtml(unsafe) {
  return unsafe.replace(/[&<"'>]/g, function(m) {
    switch (m) {
      case '&': return '&amp;';
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '"': return '&quot;';
      case "'": return '&#039;';
      default: return m;
    }
  });
}

// create regex-safe escape
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// highlight function: terms is [[term, weight], ...], majoritySign true => positive = support
function highlightText(original, terms, majorityIsGenuine) {
  if (!terms || terms.length === 0) {
    // return escaped original
    return "<p>" + escapeHtml(original) + "</p>";
  }
  // create array of objects {term, weight}
  const items = terms.map(t => ({term: t[0], weight: parseFloat(t[1])}));
  // sort by length desc to avoid partial matches
  items.sort((a,b) => b.term.length - a.term.length);
  // build pattern of alternation (escape regex)
  const patterns = items.map(it => escapeRegex(it.term));
  const re = new RegExp("(" + patterns.join("|") + ")", "gi");

  // compute max abs weight for scaling
  const maxW = Math.max(...items.map(it => Math.abs(it.weight))) || 1;

  // replace
  const replaced = escapeHtml(original).replace(re, function(matched) {
    // find item (case-insensitively)
    const found = items.find(it => it.term.toLowerCase() === matched.toLowerCase());
    const w = found ? found.weight : 0;
    const norm = Math.min(Math.abs(w) / maxW, 1.0);
    // color: positive => green (support majority), negative => red (oppose)
    // But sign interpretation: we assume weights returned are for majority label (positive supports majority)
    const isPositive = w >= 0;
    const bgAlpha = 0.2 + 0.6 * norm; // 0.2..0.8
    let bgColor = isPositive ? `rgba(16,185,129,${bgAlpha})` : `rgba(239,68,68,${bgAlpha})`;
    // border color darker
    let border = isPositive ? 'rgba(16,185,129,0.9)' : 'rgba(239,68,68,0.9)';
    return `<span class="lime-highlight" style="background:${bgColor}; border:1px solid ${border}">${matched}</span>`;
  });

  return `<div style="line-height:1.6">${replaced}</div>`;
}

// Update metrics table from server-rendered JSON inserted into DOM by Jinja
function populateMetrics(metrics) {
  const container = document.getElementById("metricsTable");
  if (!metrics || Object.keys(metrics).length === 0) {
    container.innerHTML = '<div class="text-sm text-slate-600">No metrics found (train models first)</div>';
    return;
  }
  let html = '<table class="min-w-full text-sm"><thead class="bg-slate-50"><tr><th class="p-2">Model</th><th class="p-2">Acc</th><th class="p-2">Prec</th><th class="p-2">Rec</th><th class="p-2">F1</th></tr></thead><tbody>';
  for (const [name, m] of Object.entries(metrics)) {
    html += `<tr><td class="p-2">${name}</td>
             <td class="p-2">${(m.accuracy*100).toFixed(2)}%</td>
             <td class="p-2">${(m.precision*100).toFixed(2)}%</td>
             <td class="p-2">${(m.recall*100).toFixed(2)}%</td>
             <td class="p-2">${(m.f1*100).toFixed(2)}%</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function setGauge(pct) {
  const gauge = document.getElementById("gauge");
  const label = document.getElementById("gaugeLabel");
  gauge.style.setProperty('--p', pct + '%');
  // color scale
  let col = pct > 66 ? '#10b981' : (pct > 33 ? '#f59e0b' : '#ef4444');
  gauge.style.setProperty('--color', col);
  label.innerText = pct + '%';
}

function addHistoryRow(reviewText, majLabel, consensus, avgTrust) {
  const body = document.getElementById("historyBody");
  const tr = document.createElement("tr");
  const snippet = reviewText.length > 100 ? reviewText.slice(0,100) + '…' : reviewText;
  tr.innerHTML = `<td class="p-2">${escapeHtml(snippet)}</td>
                  <td class="p-2">${majLabel}</td>
                  <td class="p-2">${consensus}%</td>
                  <td class="p-2">${avgTrust}%</td>`;
  body.prepend(tr);
}

// main handler
document.addEventListener("DOMContentLoaded", function() {
  // particles
  particlesJS("particles-js", {
    particles: { number: { value: 40, density: { enable: true, value_area: 800 } }, color: { value: "#6366f1" },
      shape: { type: "circle" }, opacity: { value: 0.12 }, size: { value: 4, random: true }, line_linked: { enable: true, distance: 150, color: "#6366f1", opacity: 0.08, width: 1 }, move: { enable: true, speed: 2 } },
    interactivity: { detect_on: "canvas", events: { onhover: { enable: true, mode: "grab" } } },
    retina_detect: true
  });

  // populate metrics (injected by Jinja into page)
  try {
    const metricsJson = window.SERVER_METRICS || {};
    populateMetrics(metricsJson);
  } catch (e) { console.warn(e); }

  // form handler
  const form = document.getElementById("reviewForm");
  form.addEventListener("submit", async function(evt) {
    evt.preventDefault();
    const loading = document.getElementById("loading");
    loading.classList.remove("hidden");
    const formData = new FormData(form);

    const resp = await fetch("/predict", { method: "POST", body: formData });
    const data = await resp.json();
    loading.classList.add("hidden");
    if (data.error) { alert(data.error); return; }

    // show sections
    document.getElementById("resultsSection").classList.remove("hidden");
    document.getElementById("explainSection").classList.remove("hidden");

    // per-model cards
    const modelGrid = document.getElementById("modelGrid");
    modelGrid.innerHTML = "";
    const labels = [], confidences = [];
    const votesCount = { "Genuine": 0, "Fake": 0 };

    for (const [model, info] of Object.entries(data.per_model)) {
      labels.push(model);
      confidences.push(info.confidence);
      votesCount[info.label] = (votesCount[info.label] || 0) + 1;
      const card = document.createElement("div");
      card.className = "p-3 rounded-lg bg-slate-50 border";
      card.innerHTML = `<div class="flex items-center justify-between"><div class="font-semibold">${model}</div>
                        <div class="${info.label==='Genuine' ? 'text-emerald-600' : 'text-rose-600'}">${info.label}</div></div>
                        <div class="mt-2 h-2 bg-slate-100 rounded overflow-hidden"><div class="${info.label==='Genuine' ? 'bg-emerald-500' : 'bg-rose-500'} h-2" style="width:${info.confidence}%"></div></div>
                        <div class="mt-1 text-xs text-slate-600">Confidence: <b>${info.confidence}%</b></div>`;
      modelGrid.appendChild(card);
    }

    // majority and bar
    const maj = data.majority_vote;
    document.getElementById("majorityLine").innerHTML = `<b>${maj.label}</b> · Consensus: <b>${maj.consensus_strength}%</b>`;
    document.getElementById("ensembleDecision").innerText = maj.label;
    document.getElementById("avgTrust").innerText = maj.avg_trust_winner;
    document.getElementById("consensusBar").style.width = maj.consensus_strength + "%";
    setGauge(maj.trust_score);

    // history add
    addHistoryRow(data.original_review, maj.label, maj.consensus_strength, maj.avg_trust_winner);

    // charts
    if (barChart) { barChart.destroy(); }
    const ctxBar = document.getElementById("barChart").getContext("2d");
    barChart = new Chart(ctxBar, {
      type: "bar",
      data: { labels: labels, datasets: [{ label: "Confidence %", data: confidences, backgroundColor: ['#6366f1','#10b981','#f59e0b','#ef4444'] }] },
      options: { scales: { y: { beginAtZero: true, max: 100 } } }
    });

    if (pieChart) { pieChart.destroy(); }
    const ctxPie = document.getElementById("pieChart").getContext("2d");
    pieChart = new Chart(ctxPie, {
      type: "pie",
      data: { labels: ["Genuine", "Fake"], datasets: [{ data: [votesCount["Genuine"]||0, votesCount["Fake"]||0], backgroundColor: ['#b91010ff','#44ef75ff'] }] }
    });

    // trend chart
    trendLabels.push(new Date().toLocaleTimeString());
    trendValues.push(maj.consensus_strength);
    if (trendChart) { trendChart.destroy(); }
    trendChart = new Chart(document.getElementById("trendChart").getContext("2d"), {
      type: "line",
      data: { labels: trendLabels, datasets: [{ label: "Consensus %", data: trendValues, borderColor: "#6366f1", fill:false, tension: 0.25 }] },
      options: { scales: { y: { beginAtZero:true, max: 100 } } }
    });

    // heuristics list
    const heurList = document.getElementById("heuristicsList");
    heurList.innerHTML = "";
    const heur = data.heuristics;
    for (const [k,v] of Object.entries(heur)) {
      const li = document.createElement("li");
      li.innerHTML = `<b>${k.replaceAll("_"," ")}</b>: ${v}`;
      heurList.appendChild(li);
    }

    // LIME highlighted text
    const limeTerms = data.lime_terms; // list of [term, weight]
    // create highlighted HTML
    const highlighted = highlightText(data.original_review, limeTerms, data.majority_vote.label === "Genuine");
    document.getElementById("highlightedText").innerHTML = highlighted;

    // show LIME term chips
    const limeList = document.getElementById("limeList");
    limeList.innerHTML = "";
    const maxAbs = Math.max(...limeTerms.map(t => Math.abs(t[1])) );
    limeTerms.forEach(([term, weight]) => {
      const w = parseFloat(weight);
      const chip = document.createElement("div");
      const color = w >= 0 ? '#10b981' : '#ef4444';
      const alpha = 0.15 + 0.65 * (Math.min(Math.abs(w) / maxAbs, 1));
      chip.className = "px-3 py-1 rounded-full text-sm";
      chip.style.background = (w>=0) ? `rgba(16,185,129,${alpha})` : `rgba(239,68,68,${alpha})`;
      chip.style.border = `1px solid ${ (w>=0)?'rgba(16,185,129,0.4)':'rgba(239,68,68,0.4)' }`;
      chip.innerText = `${term} (${w.toFixed(3)})`;
      limeList.appendChild(chip);
    });

    // Save last result for download
    window.LAST_RESULT = data;
  });

  // Download report button
  document.getElementById("downloadReport").addEventListener("click", function(){
    if (!window.LAST_RESULT) { alert("Analyze a review first."); return; }
    const res = window.LAST_RESULT;
    const report = {
      review: res.original_review,
      per_model: res.per_model,
      majority: res.majority_vote,
      heuristics: res.heuristics,
      lime_terms: res.lime_terms
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "review_report.json"; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  });
});
