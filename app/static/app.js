let sessionId = null;
let selectedFiles = [];

const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadWarnings = document.getElementById("upload-warnings");
const sessionIdEl = document.getElementById("session-id");
const sessionMetaEl = document.getElementById("session-meta");
const fileInput = document.getElementById("file-input");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const uploadProgressText = document.getElementById("upload-progress-text");
const indexProgressLabel = document.getElementById("index-progress-label");
const indexProgressBar = document.getElementById("index-progress-bar");
const indexProgressText = document.getElementById("index-progress-text");

const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const answerText = document.getElementById("answer-text");
const sourcesList = document.getElementById("sources-list");
const chartContainer = document.getElementById("chart-container");

const vizCanvas = document.getElementById("viz-canvas");
let chartInstance = null;
let indexPoller = null;
let indexPollerSession = null;
let indexPollerToken = 0;

function setStatus(message, isError = false) {
  uploadStatus.textContent = message;
  uploadStatus.style.color = isError ? "#a24030" : "#0d9488";
}

function setProgress(bar, textEl, value) {
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  if (bar) bar.style.width = `${clamped}%`;
  if (textEl) textEl.textContent = `${clamped}%`;
}

function resetProgress() {
  setProgress(uploadProgressBar, uploadProgressText, 0);
  setProgress(indexProgressBar, indexProgressText, 0);
  if (indexProgressLabel) indexProgressLabel.textContent = "Indexing";
}

function generateSessionId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID().replace(/-/g, "");
  }
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function stopIndexPolling() {
  if (indexPoller) {
    clearInterval(indexPoller);
    indexPoller = null;
  }
  indexPollerSession = null;
  indexPollerToken += 1;
}

async function pollIndexProgress(currentSessionId, pollToken) {
  if (pollToken !== indexPollerToken || currentSessionId !== indexPollerSession) return;
  try {
    const response = await fetch(`/api/progress/${currentSessionId}`);
    if (pollToken !== indexPollerToken || currentSessionId !== indexPollerSession) return;
    if (!response.ok) return;
    const data = await response.json();
    if (pollToken !== indexPollerToken || currentSessionId !== indexPollerSession) return;
    if (typeof data.percent === "number") {
      setProgress(indexProgressBar, indexProgressText, data.percent);
    }
    if (indexProgressLabel) {
      if (data.stage === "embedding") {
        indexProgressLabel.textContent = "Embedding...";
      } else if (data.stage === "completed") {
        indexProgressLabel.textContent = "Completed";
      } else {
        indexProgressLabel.textContent = "Indexing";
      }
    }
    if (data.stage === "completed" || data.stage === "error") {
      stopIndexPolling();
    }
  } catch (error) {
    if (pollToken === indexPollerToken && currentSessionId === indexPollerSession) {
      stopIndexPolling();
    }
  }
}

function startIndexPolling(currentSessionId) {
  stopIndexPolling();
  indexPollerSession = currentSessionId;
  const pollToken = indexPollerToken;
  indexPoller = setInterval(() => {
    pollIndexProgress(currentSessionId, pollToken);
  }, 800);
}

function renderWarnings(warnings) {
  uploadWarnings.innerHTML = "";
  if (!warnings || warnings.length === 0) return;
  warnings.forEach((warning) => {
    const p = document.createElement("p");
    p.textContent = warning;
    uploadWarnings.appendChild(p);
  });
}


function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsText(file);
  });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values.map((value) => value.trim());
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) return { headers: [], rows: [] };
  const headers = splitCsvLine(lines[0]);
  const rows = lines.slice(1).map(splitCsvLine);
  return { headers, rows };
}

function buildHistogram(values, bins = 8) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return { labels: [`${min.toFixed(2)}`], counts: [values.length] };
  const size = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  values.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor((value - min) / size));
    counts[index] += 1;
  });
  const labels = counts.map((_, idx) => {
    const start = min + idx * size;
    const end = start + size;
    return `${start.toFixed(1)}-${end.toFixed(1)}`;
  });
  return { labels, counts };
}

function buildTopWords(text, limit = 8) {
  const words = text.toLowerCase().split(/[^a-z0-9]+/).filter((word) => word.length > 3);
  const counts = new Map();
  words.forEach((word) => {
    counts.set(word, (counts.get(word) || 0) + 1);
  });
  const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const top = sorted.slice(0, limit);
  return {
    labels: top.map((entry) => entry[0]),
    counts: top.map((entry) => entry[1]),
  };
}

function drawBarChart({ labels, values, title }) {
  if (!vizCanvas) return;
  
  if (chartInstance) {
    chartInstance.destroy();
  }

  if (!values.length) return;

  chartInstance = new Chart(vizCanvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: title,
        data: values,
        backgroundColor: 'rgba(99, 102, 241, 0.85)',
        hoverBackgroundColor: 'rgba(79, 70, 229, 1)',
        borderRadius: 6,
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: title,
          font: { family: 'Inter', size: 16, weight: 600 },
          color: '#1e293b'
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleFont: { family: 'Inter', size: 13 },
          bodyFont: { family: 'Inter', size: 13 },
          padding: 12,
          cornerRadius: 8,
        }
      },
      scales: {
        y: { 
          beginAtZero: true,
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          ticks: { font: { family: 'Inter' }, color: '#64748b' }
        },
        x: {
          grid: { display: false },
          ticks: { 
            font: { family: 'Inter' }, 
            color: '#64748b',
            maxRotation: 45,
            minRotation: 0
          }
        }
      }
    }
  });
}

function drawChartSpec(spec) {
  if (!vizCanvas || !spec) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const palette = [
    "#6366f1", "#22c55e", "#f97316", "#0ea5e9",
    "#a855f7", "#facc15", "#ef4444", "#14b8a6",
    "#f43f5e", "#84cc16", "#fb923c", "#38bdf8",
  ];
  const getColor = (i) => palette[i % palette.length];
  const colors = spec.labels.map((_, i) => getColor(i));

  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, labels: { font: { family: "Inter", size: 12 }, color: "#334155" } },
      title: {
        display: true,
        text: spec.title,
        font: { family: "Inter", size: 16, weight: 600 },
        color: "#1e293b",
        padding: { bottom: 12 },
      },
      tooltip: {
        backgroundColor: "rgba(15,23,42,0.92)",
        titleFont: { family: "Inter", size: 13 },
        bodyFont: { family: "Inter", size: 13 },
        padding: 12,
        cornerRadius: 8,
      },
    },
  };

  const scaleXY = {
    y: {
      beginAtZero: true,
      grid: { color: "rgba(226,232,240,0.6)" },
      ticks: { font: { family: "Inter" }, color: "#64748b" },
    },
    x: {
      grid: { display: false },
      ticks: { font: { family: "Inter" }, color: "#64748b", maxRotation: 45, minRotation: 0 },
    },
  };

  const type = spec.type;

  // ── Donut ──────────────────────────────────────────────────────────────────
  if (type === "donut") {
    chartInstance = new Chart(vizCanvas, {
      type: "doughnut",
      data: {
        labels: spec.labels,
        datasets: [{ label: spec.title, data: spec.values, backgroundColor: colors, borderWidth: 2 }],
      },
      options: { ...baseOpts, cutout: "60%" },
    });
    return;
  }

  // ── Pie ────────────────────────────────────────────────────────────────────
  if (type === "pie") {
    chartInstance = new Chart(vizCanvas, {
      type: "pie",
      data: {
        labels: spec.labels,
        datasets: [{ label: spec.title, data: spec.values, backgroundColor: colors, borderWidth: 2 }],
      },
      options: baseOpts,
    });
    return;
  }

  // ── Radar ──────────────────────────────────────────────────────────────────
  if (type === "radar") {
    chartInstance = new Chart(vizCanvas, {
      type: "radar",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: spec.values,
          backgroundColor: "rgba(99,102,241,0.2)",
          borderColor: "#6366f1",
          borderWidth: 2,
          pointBackgroundColor: "#6366f1",
        }],
      },
      options: baseOpts,
    });
    return;
  }

  // ── Area ───────────────────────────────────────────────────────────────────
  if (type === "area") {
    chartInstance = new Chart(vizCanvas, {
      type: "line",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: spec.values,
          backgroundColor: "rgba(99,102,241,0.25)",
          borderColor: "#6366f1",
          borderWidth: 2,
          fill: true,
          tension: 0.35,
          pointRadius: 3,
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Line ───────────────────────────────────────────────────────────────────
  if (type === "line") {
    chartInstance = new Chart(vizCanvas, {
      type: "line",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: spec.values,
          backgroundColor: "rgba(99,102,241,0.1)",
          borderColor: "#6366f1",
          borderWidth: 2,
          fill: false,
          tension: 0.3,
          pointRadius: 3,
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Scatter ────────────────────────────────────────────────────────────────
  if (type === "scatter") {
    // values may be [x, y] arrays or plain floats (paired by index)
    let points;
    if (Array.isArray(spec.values[0])) {
      points = spec.values.map(([x, y]) => ({ x, y }));
    } else {
      points = spec.values.map((v, i) => ({ x: i, y: v }));
    }
    chartInstance = new Chart(vizCanvas, {
      type: "scatter",
      data: {
        datasets: [{
          label: spec.title,
          data: points,
          backgroundColor: "rgba(99,102,241,0.7)",
          pointRadius: 5,
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Bubble ─────────────────────────────────────────────────────────────────
  if (type === "bubble") {
    let bubbles;
    if (Array.isArray(spec.values[0]) && spec.values[0].length >= 3) {
      bubbles = spec.values.map(([x, y, r]) => ({ x, y, r: Math.max(3, r) }));
    } else if (Array.isArray(spec.values[0])) {
      bubbles = spec.values.map(([x, y], i) => ({ x, y, r: 8 }));
    } else {
      bubbles = spec.values.map((v, i) => ({ x: i, y: v, r: 8 }));
    }
    chartInstance = new Chart(vizCanvas, {
      type: "bubble",
      data: {
        datasets: [{
          label: spec.title,
          data: bubbles,
          backgroundColor: colors.map((c) => c + "aa"),
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Funnel (horizontal bar, sorted descending) ─────────────────────────────
  if (type === "funnel") {
    const paired = spec.labels.map((l, i) => ({ label: l, value: spec.values[i] }));
    paired.sort((a, b) => b.value - a.value);
    chartInstance = new Chart(vizCanvas, {
      type: "bar",
      data: {
        labels: paired.map((p) => p.label),
        datasets: [{
          label: spec.title,
          data: paired.map((p) => p.value),
          backgroundColor: paired.map((_, i) => getColor(i)),
          borderRadius: 4,
          borderWidth: 0,
        }],
      },
      options: {
        ...baseOpts,
        indexAxis: "y",
        scales: {
          x: { ...scaleXY.x, grid: { color: "rgba(226,232,240,0.6)" } },
          y: { ...scaleXY.y, grid: { display: false } },
        },
      },
    });
    return;
  }

  // ── Waterfall (floating bar segments) ─────────────────────────────────────
  if (type === "waterfall") {
    let running = 0;
    const floatData = spec.values.map((v) => {
      const seg = [running, running + v];
      running += v;
      return seg;
    });
    const barColors = spec.values.map((v) =>
      v >= 0 ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)"
    );
    chartInstance = new Chart(vizCanvas, {
      type: "bar",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: floatData,
          backgroundColor: barColors,
          borderRadius: 3,
          borderWidth: 0,
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Stacked bar ───────────────────────────────────────────────────────────
  if (type === "stacked_bar") {
    chartInstance = new Chart(vizCanvas, {
      type: "bar",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: spec.values,
          backgroundColor: colors,
          borderRadius: 4,
          borderWidth: 0,
        }],
      },
      options: {
        ...baseOpts,
        scales: {
          ...scaleXY,
          x: { ...scaleXY.x, stacked: true },
          y: { ...scaleXY.y, stacked: true },
        },
      },
    });
    return;
  }

  // ── Grouped bar ───────────────────────────────────────────────────────────
  if (type === "grouped_bar") {
    chartInstance = new Chart(vizCanvas, {
      type: "bar",
      data: {
        labels: spec.labels,
        datasets: [{
          label: spec.title,
          data: spec.values,
          backgroundColor: colors,
          borderRadius: 4,
          borderWidth: 0,
        }],
      },
      options: { ...baseOpts, scales: scaleXY },
    });
    return;
  }

  // ── Histogram / Bar (default) ─────────────────────────────────────────────
  chartInstance = new Chart(vizCanvas, {
    type: "bar",
    data: {
      labels: spec.labels,
      datasets: [{
        label: spec.title,
        data: spec.values,
        backgroundColor: "rgba(99,102,241,0.85)",
        hoverBackgroundColor: "rgba(79,70,229,1)",
        borderRadius: 6,
        borderWidth: 0,
      }],
    },
    options: { ...baseOpts, scales: scaleXY },
  });
}


if (fileInput) {
  const fileLabelText = document.querySelector(".file-label-text");
  fileInput.addEventListener("change", () => {
    selectedFiles = Array.from(fileInput.files || []);
    if (selectedFiles.length > 0) {
      if (selectedFiles.length === 1) {
        fileLabelText.textContent = selectedFiles[0].name;
      } else {
        fileLabelText.textContent = `${selectedFiles.length} files selected`;
      }
    } else {
      fileLabelText.textContent = "Choose files or drag & drop";
    }
  });
}

window.addEventListener("resize", () => {
  if (chartInstance) {
    chartInstance.resize();
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setStatus("Please select one or more files.", true);
    return;
  }

  setStatus("Uploading and indexing...", false);
  renderWarnings([]);
  resetProgress();

  if (!sessionId) {
    sessionId = generateSessionId();
  }

  const formData = new FormData();
  Array.from(fileInput.files).forEach((file) => formData.append("files", file));
  formData.append("session_id", sessionId);

  startIndexPolling(sessionId);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload", true);
  xhr.responseType = "json";

  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      const percent = (event.loaded / event.total) * 100;
      setProgress(uploadProgressBar, uploadProgressText, percent);
    }
  };

  xhr.onload = () => {
    let data = xhr.response;
    if (!data && xhr.responseText) {
      try {
        data = JSON.parse(xhr.responseText);
      } catch (parseError) {
        data = {};
      }
    }
    if (!data) {
      data = {};
    }
    if (xhr.status < 200 || xhr.status >= 300) {
      const message = data.detail || "Upload failed.";
      setStatus(message, true);
      stopIndexPolling();
      return;
    }

    setProgress(uploadProgressBar, uploadProgressText, 100);
    setProgress(indexProgressBar, indexProgressText, 100);
    stopIndexPolling();

    sessionId = data.session_id;
    sessionIdEl.textContent = sessionId;
    const fileCount = Array.isArray(data.files) ? data.files.length : data.files;
    const chunksTotal = data.chunks_total ?? data.chunks;
    const chunksAdded = data.chunks_added ?? data.chunks;
    if (data.appended) {
      sessionMetaEl.textContent = `${fileCount} files added, ${chunksTotal} total chunks.`;
    } else {
      sessionMetaEl.textContent = `${fileCount} files indexed, ${chunksAdded} chunks created.`;
    }
    setStatus("Index ready. Ask a question.");
    renderWarnings(data.warnings);
    selectedFiles = Array.from(fileInput.files || []);
  };

  xhr.onerror = () => {
    setStatus("Upload failed.", true);
    stopIndexPolling();
  };

  xhr.send(formData);
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!sessionId) {
    answerText.textContent = "Upload documents to start a session.";
    return;
  }

  const question = questionInput.value.trim();
  if (!question) {
    answerText.textContent = "Enter a question first.";
    return;
  }

  answerText.textContent = "Thinking...";
  sourcesList.textContent = "Searching sources...";
  // Hide previous chart while the new answer loads
  if (chartContainer) chartContainer.style.display = "none";
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId, question }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Question failed.");
    }

    answerText.textContent = (data.answer || "No answer returned.")
      .replace(/\[[\w.:]+\]/g, "")
      .trim();
    if (data.sources && data.sources.length) {
      sourcesList.innerHTML = "";
      data.sources.forEach((source) => {
        const item = document.createElement("div");
        item.textContent = `${source.file_name} (chunk ${source.chunk_index})`; 
        sourcesList.appendChild(item);
      });
    } else {
      sourcesList.textContent = "No sources returned.";
    }

    if (data.chart_spec) {
      if (chartContainer) {
        chartContainer.style.display = "block";
      }
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          drawChartSpec(data.chart_spec);
        });
      });
    } else if (chartContainer) {
      chartContainer.style.display = "none";
    }
  } catch (error) {
    answerText.textContent = error.message;
    sourcesList.textContent = "";
  }
});
