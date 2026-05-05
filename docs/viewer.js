// Project-page interactive viewer.
// Loads docs/data/annotations.json and lets the reader compare three
// annotation layers per image: original KITTI labels, detector predictions,
// and the authors' re-annotated ground truth.

const REPO_URL = "https://github.com/mos-ks/MUHA";

const COLORS = {
  kitti:       { stroke: "#b31529", fill: "rgba(179,21,41,0.10)", dash: [6, 4] },
  predictions: { stroke: "#1065ab", fill: "rgba(16,101,171,0.10)", dash: [] },
  relabeled:   { stroke: "#2c8a3d", fill: "rgba(44,138,61,0.12)", dash: [] },
};

// Map a normalized uncertainty in [0, 1] onto the same blue→white→red scale
// as the labeling tool, returning a stroke colour.
function uncertaintyToColor(unc) {
  const t = Math.max(0, Math.min(1, unc));
  let r, g, b;
  if (t <= 0.5) {
    const k = t * 2;
    r = Math.round(16  + k * (249 - 16));
    g = Math.round(101 + k * (249 - 101));
    b = Math.round(171 + k * (249 - 171));
  } else {
    const k = (t - 0.5) * 2;
    r = Math.round(249 + k * (179 - 249));
    g = Math.round(249 + k * (21  - 249));
    b = Math.round(249 + k * (41  - 249));
  }
  return `rgb(${r},${g},${b})`;
}

const state = {
  bundle: [],
  filtered: [],
  binIndex: { low: new Set(), mid: new Set(), high: new Set() },
  index: 0,
  bin: "all",
  layers: { kitti: true, predictions: true, relabeled: true },
  colorByUncertainty: false,
  imageCache: new Map(),
};

const els = {};

function $(sel) { return document.querySelector(sel); }

function setupRepoLinks() {
  document.querySelectorAll("#repo-link, #repo-link-2").forEach((a) => {
    a.href = REPO_URL;
  });
}

async function loadAnnotations() {
  // Don't pin to "force-cache": when the JSON is regenerated, returning
  // visitors would otherwise keep seeing stale annotations.
  const r = await fetch("data/annotations.json");
  if (!r.ok) throw new Error(`Failed to load annotations: ${r.status}`);
  return r.json();
}

async function loadBins() {
  const fetchList = async (name) => {
    try {
      const r = await fetch(`data/${name}_uncertainty.txt`);
      if (!r.ok) return [];
      const txt = await r.text();
      return txt.split(/\s+/).filter(Boolean);
    } catch { return []; }
  };
  const [low, mid, high] = await Promise.all(["low", "mid", "high"].map(fetchList));
  state.binIndex.low  = new Set(low);
  state.binIndex.mid  = new Set(mid);
  state.binIndex.high = new Set(high);
}

function applyBinFilter() {
  if (state.bin === "all") {
    state.filtered = state.bundle.slice();
  } else {
    state.filtered = state.bundle.filter((img) => state.binIndex[state.bin].has(img.id));
  }
  if (state.filtered.length === 0) state.filtered = state.bundle.slice();
  state.index = 0;
}

function preload(src) {
  if (state.imageCache.has(src)) return Promise.resolve(state.imageCache.get(src));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => { state.imageCache.set(src, img); resolve(img); };
    img.onerror = reject;
    img.src = src;
  });
}

function drawBox(ctx, box, color, dash) {
  const [x1, y1, x2, y2] = box;
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.setLineDash(dash || []);
  ctx.strokeStyle = color.stroke || color;
  if (color.fill) {
    ctx.fillStyle = color.fill;
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.restore();
}

function labelBox(ctx, box, text, color) {
  const [x1, y1] = box;
  ctx.save();
  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  const padding = 4;
  const w = ctx.measureText(text).width + padding * 2;
  const h = 18;
  ctx.fillStyle = color;
  ctx.fillRect(x1 - 1, Math.max(0, y1 - h), w, h);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x1 - 1 + padding, Math.max(h - 5, y1 - 5));
  ctx.restore();
}

async function render() {
  const item = state.filtered[state.index];
  if (!item) return;
  const img = await preload(item.image);

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  els.canvas.width = item.width * dpr;
  els.canvas.height = item.height * dpr;
  els.canvas.style.aspectRatio = `${item.width} / ${item.height}`;
  const ctx = els.canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, item.width, item.height);

  if (state.layers.kitti) {
    for (const b of item.kitti) {
      drawBox(ctx, b.bbox, COLORS.kitti, COLORS.kitti.dash);
    }
  }
  if (state.layers.predictions) {
    for (const b of item.predictions) {
      let color = COLORS.predictions;
      if (state.colorByUncertainty) {
        // The uncertainties are roughly in 0–5 in the file; clip to that range.
        const norm = Math.min(1, b.uncertainty_mean / 5);
        const stroke = uncertaintyToColor(norm);
        color = { stroke, fill: "rgba(0,0,0,0)" };
      }
      drawBox(ctx, b.bbox, color);
    }
  }
  if (state.layers.relabeled) {
    for (const b of item.relabeled) {
      drawBox(ctx, b.bbox, COLORS.relabeled);
    }
  }

  els.imgId.textContent = `KITTI #${item.id}`;
  els.imgPos.textContent = `${state.index + 1} / ${state.filtered.length}`;

  const counts = [];
  if (state.layers.kitti)       counts.push(`${item.kitti.length} original`);
  if (state.layers.predictions) counts.push(`${item.predictions.length} predicted`);
  if (state.layers.relabeled)   counts.push(`${item.relabeled.length} relabeled`);
  els.meta.textContent = `${item.width}×${item.height}  ·  ${counts.join("  ·  ")}`;

  els.prevBtn.disabled = state.index === 0;
  els.nextBtn.disabled = state.index === state.filtered.length - 1;

  // Pre-warm next/prev images so navigation feels instant.
  const next = state.filtered[state.index + 1];
  const prev = state.filtered[state.index - 1];
  if (next) preload(next.image);
  if (prev) preload(prev.image);
}

function safeRender() {
  // Wrap render() so a single failed image doesn't leave the canvas-meta
  // banner stuck on a stale error message after layer/bin/nav changes.
  return render().catch((err) => {
    console.error(err);
    els.meta.textContent = `Failed to render image: ${err?.message || err?.type || err}`;
  });
}

function go(delta) {
  const next = state.index + delta;
  if (next < 0 || next >= state.filtered.length) return;
  state.index = next;
  safeRender();
}

function bindUi() {
  els.canvas = $("#annotation-canvas");
  els.meta = $("#canvas-meta");
  els.imgId = $("#img-id");
  els.imgPos = $("#img-pos");
  els.prevBtn = $("#prev-btn");
  els.nextBtn = $("#next-btn");

  els.prevBtn.addEventListener("click", () => go(-1));
  els.nextBtn.addEventListener("click", () => go(1));

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
  });

  document.querySelectorAll('.layer-toggles input[data-layer]').forEach((cb) => {
    cb.addEventListener("change", () => {
      state.layers[cb.dataset.layer] = cb.checked;
      safeRender();
    });
  });
  $("#color-by-uncertainty").addEventListener("change", (e) => {
    state.colorByUncertainty = e.target.checked;
    safeRender();
  });

  document.querySelectorAll('.bin-toggles input[name="bin"]').forEach((rb) => {
    rb.addEventListener("change", () => {
      if (!rb.checked) return;
      state.bin = rb.value;
      applyBinFilter();
      safeRender();
    });
  });

  $("#copy-bib").addEventListener("click", async (e) => {
    const bib = $(".bib code").innerText;
    try {
      await navigator.clipboard.writeText(bib);
      const original = e.target.textContent;
      e.target.textContent = "Copied ✓";
      setTimeout(() => { e.target.textContent = original; }, 1500);
    } catch {
      e.target.textContent = "Copy failed";
    }
  });
}

(async function main() {
  setupRepoLinks();
  bindUi();
  try {
    const [bundle] = await Promise.all([loadAnnotations(), loadBins()]);
    state.bundle = bundle;
    applyBinFilter();
    await render();
  } catch (err) {
    console.error(err);
    $("#canvas-meta").textContent = "Failed to load annotations.";
  }
})();
