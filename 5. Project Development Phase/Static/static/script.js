/* =========================================================================
   HDI Atlas — frontend logic
   Handles: prediction form (AJAX), gauge animation, dark/light mode,
   prediction history table + distribution chart, batch CSV upload.
   ========================================================================= */

document.getElementById("year").textContent = new Date().getFullYear();

/* ---------------- Theme toggle ---------------- */
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeIcon.textContent = theme === "light" ? "\u2600" : "\u263D"; // sun / moon
}
let currentTheme = "dark";
applyTheme(currentTheme);
themeToggle.addEventListener("click", () => {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(currentTheme);
});

/* ---------------- Category -> gauge mapping ---------------- */
const CATEGORY_ORDER = ["Low", "Medium", "High", "Very High"];
const CATEGORY_COLOR_VAR = {
  "Low": "--low",
  "Medium": "--medium",
  "High": "--high",
  "Very High": "--very-high",
};
const CATEGORY_CLASS = {
  "Low": "cat-low",
  "Medium": "cat-medium",
  "High": "cat-high",
  "Very High": "cat-very-high",
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setGauge(category) {
  const idx = CATEGORY_ORDER.indexOf(category);
  const fraction = (idx + 1) / CATEGORY_ORDER.length; // 0.25, 0.5, 0.75, 1.0
  const arc = document.getElementById("gaugeArc");
  const needle = document.getElementById("gaugeNeedle");

  const circumference = 283; // approx length of the semicircle path
  arc.style.stroke = cssVar(CATEGORY_COLOR_VAR[category]);
  arc.style.strokeDashoffset = String(circumference * (1 - fraction));

  // Needle sweeps from -90deg (left) to +90deg (right) across the semicircle
  const angle = -90 + fraction * 180;
  needle.setAttribute("transform", `rotate(${angle} 110 110)`);
}

/* ---------------- Prediction history (client-side render) ---------------- */
let sessionHistory = [];

function renderHistoryTable() {
  const body = document.getElementById("historyBody");
  body.innerHTML = "";
  sessionHistory.slice().reverse().forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.timestamp}</td>
      <td>${r.inputs["Life Expectancy"]}</td>
      <td>${r.inputs["Mean Years of Schooling"]}</td>
      <td>${r.inputs["Expected Years of Schooling"]}</td>
      <td>${r.inputs["GNI Per Capita"]}</td>
      <td class="${CATEGORY_CLASS[r.category] || ""}">${r.category}</td>
      <td>${r.confidence !== null && r.confidence !== undefined ? r.confidence + "%" : "—"}</td>
    `;
    body.appendChild(tr);
  });
  renderDistChart();
}

function renderDistChart() {
  const svg = document.getElementById("distChart");
  svg.innerHTML = "";
  const counts = { "Low": 0, "Medium": 0, "High": 0, "Very High": 0 };
  sessionHistory.forEach((r) => { if (counts[r.category] !== undefined) counts[r.category]++; });

  const max = Math.max(1, ...Object.values(counts));
  const barWidth = 70;
  const gap = 30;
  const chartHeight = 130;
  const baseY = 150;

  CATEGORY_ORDER.forEach((cat, i) => {
    const x = 30 + i * (barWidth + gap);
    const h = (counts[cat] / max) * chartHeight;
    const y = baseY - h;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", barWidth);
    rect.setAttribute("height", h);
    rect.setAttribute("rx", 4);
    rect.setAttribute("fill", cssVar(CATEGORY_COLOR_VAR[cat]));
    rect.setAttribute("opacity", "0.88");
    svg.appendChild(rect);

    const valueText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    valueText.setAttribute("x", x + barWidth / 2);
    valueText.setAttribute("y", y - 6);
    valueText.setAttribute("text-anchor", "middle");
    valueText.setAttribute("class", "bar-value");
    valueText.textContent = counts[cat];
    svg.appendChild(valueText);

    const labelText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    labelText.setAttribute("x", x + barWidth / 2);
    labelText.setAttribute("y", baseY + 16);
    labelText.setAttribute("text-anchor", "middle");
    labelText.setAttribute("class", "bar-label");
    labelText.textContent = cat;
    svg.appendChild(labelText);
  });
}

/* ---------------- Predict form ---------------- */
const predictForm = document.getElementById("predictForm");
const predictBtn = document.getElementById("predictBtn");
const btnSpinner = document.getElementById("btnSpinner");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");

predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Clear previous field errors
  document.querySelectorAll(".field-error").forEach((el) => (el.textContent = ""));

  const formData = new FormData(predictForm);
  const payload = Object.fromEntries(formData.entries());

  predictBtn.disabled = true;
  btnSpinner.hidden = false;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.success) {
      (data.errors || ["Something went wrong."]).forEach((err) => {
        // naive mapping: show first error near the form actions
        const generic = document.querySelector('[data-for="life_expectancy"]');
        if (generic) generic.textContent = err;
      });
      return;
    }

    resultEmpty.hidden = true;
    resultContent.hidden = false;

    setGauge(data.category);
    const catEl = document.getElementById("resultCategory");
    catEl.textContent = data.category;
    catEl.className = "result-category " + (CATEGORY_CLASS[data.category] || "");

    document.getElementById("resultConfidence").textContent =
      data.confidence !== null ? `Model confidence: ${data.confidence}%` : "";
    document.getElementById("resultMeaning").textContent = data.meaning;

    const suggEl = document.getElementById("suggestions");
    if (data.suggestions && data.suggestions.length) {
      suggEl.innerHTML =
        "<h4>Suggestions for Improvement</h4><ul>" +
        data.suggestions.map((s) => `<li>${s}</li>`).join("") +
        "</ul>";
    } else {
      suggEl.innerHTML = "";
    }

    sessionHistory.push(data);
    renderHistoryTable();
  } catch (err) {
    alert("Prediction request failed. Please make sure the server is running.");
  } finally {
    predictBtn.disabled = false;
    btnSpinner.hidden = true;
  }
});

document.getElementById("resetBtn").addEventListener("click", () => {
  resultEmpty.hidden = false;
  resultContent.hidden = true;
});

/* ---------------- Batch prediction ---------------- */
const batchForm = document.getElementById("batchForm");
batchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) return;

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/batch_predict", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.success) {
      alert((data.errors || ["Batch prediction failed."]).join(" "));
      return;
    }

    document.getElementById("batchResult").hidden = false;
    document.getElementById("batchSummary").textContent =
      `Predicted ${data.rows} rows. Distribution: ` +
      Object.entries(data.distribution).map(([k, v]) => `${k}: ${v}`).join(", ");

    const table = document.getElementById("batchTable");
    if (data.preview.length) {
      const cols = Object.keys(data.preview[0]);
      let html = "<thead><tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
      data.preview.forEach((row) => {
        html += "<tr>" + cols.map((c) => `<td>${row[c]}</td>`).join("") + "</tr>";
      });
      html += "</tbody>";
      table.innerHTML = html;
    }
  } catch (err) {
    alert("Batch upload failed. Please check your CSV file and try again.");
  }
});

/* ---------------- Init ---------------- */
renderDistChart();
