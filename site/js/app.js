/**
 * Inflation Dashboard - Frontend
 * Carrega dados JSON e renderiza a interface dinamicamente.
 */

const DATA_URL = "data/brazil/historical.json";

let appData = null;
let generalSeries = [];

// Formatadores
const fmtPct = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
};

const fmtNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) return "--";
  return value.toFixed(decimals);
};

const fmtPeriod = (period) => {
  if (!period || period.length !== 6) return period;
  const year = period.slice(0, 4);
  const month = period.slice(4, 6);
  const months = [
    "jan", "fev", "mar", "abr", "mai", "jun",
    "jul", "ago", "set", "out", "nov", "dez"
  ];
  return `${months[parseInt(month, 10) - 1]}/${year.slice(2)}`;
};

const colorClass = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "neutral";
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "neutral";
};

// Carrega dados
async function loadData() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    appData = await response.json();
    generalSeries = appData.general_series || [];
    initDashboard();
  } catch (error) {
    console.error("Erro ao carregar dados:", error);
    document.querySelector("main").innerHTML = `
      <div class="card">
        <p>Não foi possível carregar os dados. Verifique o console para mais detalhes.</p>
        <p><small>${error.message}</small></p>
      </div>
    `;
  }
}

// Inicializa o dashboard
function initDashboard() {
  const periods = Object.keys(appData.historical).sort();
  const latestPeriod = periods[periods.length - 1];

  populatePeriodSelect(periods, latestPeriod);
  setupTabs();
  renderPeriod(latestPeriod);

  document.getElementById("period-select").addEventListener("change", (e) => {
    renderPeriod(e.target.value);
  });

  // Data de atualização
  const updateDate = appData.metadata?.last_update;
  if (updateDate) {
    const d = new Date(updateDate);
    document.getElementById("update-date").textContent = d.toLocaleDateString("pt-BR");
  }
}

// Popula select de períodos
function populatePeriodSelect(periods, selected) {
  const select = document.getElementById("period-select");
  select.innerHTML = "";
  [...periods].reverse().forEach((period) => {
    const option = document.createElement("option");
    option.value = period;
    option.textContent = fmtPeriod(period);
    if (period === selected) option.selected = true;
    select.appendChild(option);
  });
}

// Renderiza um período específico
function renderPeriod(period) {
  const data = appData.historical[period];
  if (!data) return;

  renderKPIs(data.resumo, period);
  renderChart(period);
  renderResumoTable(data.resumo);
  renderDestaquesTable(data.destaques);
}

// Renderiza KPIs
function renderKPIs(resumo, period) {
  const geral = resumo.find((item) => item.metric === "Índice Geral");

  if (geral) {
    document.getElementById("kpi-mom").textContent = fmtPct(geral.mom);
    document.getElementById("kpi-mom").className = `kpi-value ${colorClass(geral.mom)}`;
    document.getElementById("kpi-period").textContent = fmtPeriod(period);

    document.getElementById("kpi-yoy").textContent = fmtPct(geral.yoy);
    document.getElementById("kpi-yoy").className = `kpi-value ${colorClass(geral.yoy)}`;
    document.getElementById("kpi-yoy-period").textContent = "acumulado 12 meses";
  }

  // Maior grupo do mês (exclui índice geral)
  const groups = resumo.filter((item) => item.metric !== "Índice Geral");
  const topGroup = groups.reduce((max, item) =>
    item.mom > max.mom ? item : max, groups[0] || {});

  if (topGroup && topGroup.metric) {
    document.getElementById("kpi-top-group").textContent = topGroup.metric;
    document.getElementById("kpi-top-group-value").textContent =
      `${fmtPct(topGroup.mom)} · peso ${fmtNumber(topGroup.weight, 1)}%`;
  }
}

// Renderiza tabela resumo
function renderResumoTable(resumo) {
  const tbody = document.querySelector("#resumo-table tbody");
  tbody.innerHTML = "";

  resumo.forEach((item) => {
    const tr = document.createElement("tr");
    const isGroup = item.metric === "Índice Geral";

    tr.innerHTML = `
      <td class="text-left ${isGroup ? 'metric-group' : 'metric-name'}">${item.metric}</td>
      <td>${fmtNumber(item.weight, 2)}</td>
      <td class="${colorClass(item.mom)}">${fmtPct(item.mom)}</td>
      <td class="${colorClass(item.mom_t_1)}">${fmtPct(item.mom_t_1)}</td>
      <td class="${colorClass(item.mom_t_12)}">${fmtPct(item.mom_t_12)}</td>
      <td class="${colorClass(item.yoy)}">${fmtPct(item.yoy)}</td>
      <td class="${colorClass(item.yoy_t_1)}">${fmtPct(item.yoy_t_1)}</td>
      <td class="${colorClass(item.yoy_t_12)}">${fmtPct(item.yoy_t_12)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Renderiza tabelas de destaques
function renderDestaquesTable(destaques) {
  renderDestaqueList("destaques-positive", destaques.positive, "positive");
  renderDestaqueList("destaques-negative", destaques.negative, "negative");
}

function renderDestaqueList(elementId, items, type) {
  const tbody = document.querySelector(`#${elementId} tbody`);
  tbody.innerHTML = "";

  if (!items || items.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" class="text-left neutral">Nenhum destaque encontrado.</td>`;
    tbody.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    const badgeClass = type === "positive" ? "badge-positive" : "badge-negative";
    tr.innerHTML = `
      <td class="text-left metric-name">${item.name}</td>
      <td>${fmtNumber(item.weight, 2)}</td>
      <td class="${colorClass(item.mom)}">${fmtPct(item.mom)}</td>
      <td><span class="badge ${badgeClass}">${item.bps > 0 ? "+" : ""}${item.bps.toFixed(1)}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// Renderiza gráfico histórico
function renderChart(highlightPeriod) {
  const periods = generalSeries.map((d) => fmtPeriod(d.period));
  const momValues = generalSeries.map((d) => d.mom);
  const yoyValues = generalSeries.map((d) => d.yoy);

  const traceMoM = {
    x: periods,
    y: momValues,
    type: "bar",
    name: "Variação mensal",
    marker: { color: "#2563eb" },
    opacity: 0.85,
  };

  const traceYoY = {
    x: periods,
    y: yoyValues,
    type: "scatter",
    mode: "lines+markers",
    name: "Variação 12 meses",
    line: { color: "#dc2626", width: 2.5 },
    marker: { size: 5 },
    yaxis: "y2",
  };

  const layout = {
    autosize: true,
    height: 360,
    margin: { l: 50, r: 50, t: 20, b: 50 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", size: 12, color: "#4b5563" },
    showlegend: true,
    legend: { orientation: "h", y: -0.18 },
    xaxis: {
      tickangle: -45,
      gridcolor: "#f3f4f6",
    },
    yaxis: {
      title: "MoM (%)",
      gridcolor: "#f3f4f6",
      zerolinecolor: "#e5e7eb",
      tickformat: ",.2f",
    },
    yaxis2: {
      title: "YoY (%)",
      overlaying: "y",
      side: "right",
      showgrid: false,
      zerolinecolor: "#e5e7eb",
      tickformat: ",.2f",
    },
    annotations: [
      {
        x: fmtPeriod(highlightPeriod),
        y: generalSeries.find((d) => d.period === highlightPeriod)?.mom || 0,
        xref: "x",
        yref: "y",
        text: "Selecionado",
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1,
        arrowcolor: "#1a1d23",
        ax: 0,
        ay: -40,
        font: { size: 11, color: "#1a1d23" },
      },
    ],
  };

  const config = { responsive: true, displayModeBar: false };

  Plotly.newPlot("history-chart", [traceMoM, traceYoY], layout, config);
}

// Controle de abas
function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.getElementById(`tab-${tabId}`).classList.add("active");
    });
  });
}

// Inicia
loadData();
