// ============================================================
//  Poisson GPU Mapping Lab
//  Educational simulator: how virtual topology maps to physical
// ============================================================

// ---------- problem (constant) ----------
const N            = 1024;            // grid points per side
const ITERS        = 10000;           // Jacobi iterations
const FLOP_PER_CELL = 7;              // 5-point stencil ~7 FLOPs
const TOTAL_FLOP   = N * N * FLOP_PER_CELL * ITERS;  // ~73.4 GFLOP

// ---------- placeholder GPU specs ----------
//   sms             : streaming multiprocessors
//   maxWarpsPerSm   : warp slots per SM
//   peakTflops      : peak FLOPs / second (TFLOPs)
//   bandwidthGBs    : peak HBM bandwidth, GB/s
//   costPerHour     : USD / hour rented
//   sharedKB        : shared mem per SM (KB)
const GPUS = [
  {
    id: 'aurora',  name: 'Aurora-T1',    cls: 'entry-level',
    sms: 16,  maxWarpsPerSm: 32, peakTflops: 1.0,  bandwidthGBs: 64,   costPerHour: 0.50, sharedKB: 48
  },
  {
    id: 'nebula',  name: 'Nebula-V',     cls: 'mid-range',
    sms: 40,  maxWarpsPerSm: 48, peakTflops: 6.5,  bandwidthGBs: 320,  costPerHour: 1.50, sharedKB: 64
  },
  {
    id: 'quasar',  name: 'Quasar-X100',  cls: 'datacenter',
    sms: 108, maxWarpsPerSm: 64, peakTflops: 19.5, bandwidthGBs: 1555, costPerHour: 4.00, sharedKB: 100
  },
  {
    id: 'pulsar',  name: 'Pulsar-H200',  cls: 'hpc accelerator',
    sms: 132, maxWarpsPerSm: 64, peakTflops: 40.0, bandwidthGBs: 3350, costPerHour: 8.00, sharedKB: 228
  }
];

const SM_MAX_THREADS = 2048;   // per-SM thread cap (typical Ampere/Hopper)
const SM_MAX_BLOCKS  = 32;     // per-SM resident block cap
const WARP           = 32;

const MEM_FACTOR = {
  global:    0.22,
  pingpong:  0.42,
  shared:    0.72,
  sharedpp:  0.94,
};

const MEM_LABEL = {
  global:   'global only',
  pingpong: 'ping-pong global',
  shared:   'shared-mem tiling',
  sharedpp: 'shared + ping-pong',
};

// ---------- state ----------
let state = {
  gpu: GPUS[1],          // default Nebula-V
  bx: 16,
  by: 16,
  mem: 'pingpong',
};

let best = loadBest();

// ============================================================
//  CORE SIMULATION
// ============================================================
function simulate(s) {
  const tpb     = s.bx * s.by;
  const valid   = tpb >= 1 && tpb <= 1024;

  const gridX   = Math.ceil(N / s.bx);
  const gridY   = Math.ceil(N / s.by);
  const totalThreads = gridX * gridY * tpb;
  const warpsPerBlock = Math.ceil(tpb / WARP);

  // warp efficiency — last warp may be partial
  const warpEff = valid ? (tpb / (warpsPerBlock * WARP)) : 0;

  // coalescing — block dim X close to 32 helps row-major reads
  // simple model: full at bx>=32, linearly down to bx==1
  const coalesce = Math.min(1, s.bx / WARP) * 0.6 + 0.4 * (s.bx >= 4 ? 1 : s.bx / 4);
  const coalesceN = Math.min(1, coalesce);

  // occupancy
  const limByThreads = Math.floor(SM_MAX_THREADS / tpb);
  const limByBlocks  = SM_MAX_BLOCKS;
  // shared memory limit (assumes shared pattern uses ~ block area * 8 bytes)
  let limByShared = SM_MAX_BLOCKS;
  if (s.mem === 'shared' || s.mem === 'sharedpp') {
    const bytesPerBlock = (s.bx + 2) * (s.by + 2) * 8 * (s.mem === 'sharedpp' ? 2 : 1);
    limByShared = Math.floor((s.gpu.sharedKB * 1024) / bytesPerBlock);
  }
  const activeBlocksPerSm = Math.max(1, Math.min(limByThreads, limByBlocks, limByShared));
  const activeWarpsPerSm  = activeBlocksPerSm * warpsPerBlock;
  const occupancy = Math.min(1, activeWarpsPerSm / s.gpu.maxWarpsPerSm);

  // memory pattern factor
  const memFac = MEM_FACTOR[s.mem];

  // effective throughput (TFLOPs)
  const effTflops = s.gpu.peakTflops * occupancy * warpEff * coalesceN * memFac;

  // time to solve (seconds)
  const totalFlops = TOTAL_FLOP;                          // FLOPs
  const timeSec = totalFlops / (effTflops * 1e12);

  // tiny per-iteration launch overhead
  const overhead = ITERS * 2e-6;
  const totalSec = timeSec + overhead;

  // cost: USD
  const costUSD = (totalSec / 3600) * s.gpu.costPerHour;

  // score — higher is better, balances time x cost
  // normalize against a "good" baseline (~5ms x $1e-6)
  const score = 1000 / Math.max(1e-9, totalSec * 1000 * (costUSD * 1e6));

  // SMs used = how many we can keep busy
  const totalBlocks = gridX * gridY;
  const smUsed = Math.min(s.gpu.sms, Math.ceil(totalBlocks / activeBlocksPerSm));

  return {
    valid, tpb, gridX, gridY, totalThreads, warpsPerBlock,
    warpEff, coalesce: coalesceN, occupancy, memFac,
    effTflops, timeSec: totalSec, costUSD, score,
    activeBlocksPerSm, activeWarpsPerSm, smUsed, totalBlocks,
  };
}

// ============================================================
//  UI BINDINGS
// ============================================================
const $ = (id) => document.getElementById(id);

function renderGpus() {
  const el = $('gpuGrid');
  el.innerHTML = '';
  GPUS.forEach(g => {
    const card = document.createElement('button');
    card.className = 'gpu-card' + (g.id === state.gpu.id ? ' active' : '');
    card.dataset.id = g.id;
    card.innerHTML = `
      <div class="gpu-name">${g.name}<span class="gpu-tag">${g.id}</span></div>
      <div class="gpu-class">${g.cls}</div>
      <div class="gpu-specs">
        <span>SMs</span><b>${g.sms}</b>
        <span>warps / SM</span><b>${g.maxWarpsPerSm}</b>
        <span>peak</span><b>${g.peakTflops} TF</b>
        <span>HBM</span><b>${g.bandwidthGBs} GB/s</b>
      </div>
      <div class="gpu-cost">$${g.costPerHour.toFixed(2)} / hr</div>
    `;
    card.addEventListener('click', () => {
      state.gpu = g;
      renderGpus();
      update();
    });
    el.appendChild(card);
  });
}

function fmtTime(sec) {
  if (sec >= 1)      return sec.toFixed(2) + ' s';
  if (sec >= 1e-3)   return (sec * 1000).toFixed(2) + ' ms';
  return (sec * 1e6).toFixed(1) + ' µs';
}
function fmtCost(usd) {
  if (usd >= 1)      return '$' + usd.toFixed(2);
  if (usd >= 1e-3)   return '$' + usd.toFixed(4);
  return '$' + usd.toExponential(2);
}
function fmtThreads(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return n.toString();
}
function pct(x) { return Math.round(x * 100) + '%'; }

function renderViz(r) {
  const svg = $('vizSvg');
  const W = 400, H = 400;

  // we draw two panels side by side:
  //  left: grid of BLOCKS (gridX x gridY, scaled to fit)
  //  right: zoom of ONE block showing thread mesh (bx x by)
  const pad = 14;
  const gap = 14;
  const panelW = (W - pad * 2 - gap) / 2;
  const panelH = H - pad * 2;

  const gridX = r.gridX, gridY = r.gridY;
  const dispGX = Math.min(gridX, 32);
  const dispGY = Math.min(gridY, 32);

  let s = '';
  // background
  s += `<rect width="${W}" height="${H}" fill="#0f1014"/>`;

  // ----- LEFT: blocks grid -----
  const cellW = panelW / dispGX;
  const cellH = panelH / dispGY;
  // active blocks per SM dictates per-block "saturation"
  const occColor = (i, j) => {
    // simulate which SM hosts this block (round-robin)
    const blockIdx = j * dispGX + i;
    const sm = blockIdx % state.gpu.sms;
    // active blocks per SM "stripe"
    const slot = Math.floor(blockIdx / state.gpu.sms) % r.activeBlocksPerSm;
    const stripe = slot / r.activeBlocksPerSm;
    const alpha = 0.45 + 0.45 * (1 - stripe);
    return alpha;
  };
  for (let j = 0; j < dispGY; j++) {
    for (let i = 0; i < dispGX; i++) {
      const x = pad + i * cellW;
      const y = pad + j * cellH;
      const a = occColor(i, j);
      s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(cellW-1).toFixed(1)}" height="${(cellH-1).toFixed(1)}" fill="rgba(79,106,173,${a.toFixed(2)})"/>`;
    }
  }
  s += `<text x="${pad}" y="${pad + panelH + 9}" font-family="inherit" font-size="9" fill="#6e7280" letter-spacing="1.4">GRID &middot; ${gridX}&times;${gridY} blocks${gridX > dispGX ? ' (sampled)' : ''}</text>`;

  // ----- RIGHT: zoom of one block -----
  const px = pad + panelW + gap;
  const py = pad;
  s += `<rect x="${px}" y="${py}" width="${panelW}" height="${panelH}" fill="#0a0a0a" stroke="#1f2128"/>`;
  const tw = panelW / Math.max(1, state.bx);
  const th = panelH / Math.max(1, state.by);
  // thread cells
  for (let j = 0; j < state.by; j++) {
    for (let i = 0; i < state.bx; i++) {
      const threadIdx = j * state.bx + i;
      const warpIdx   = Math.floor(threadIdx / WARP);
      const fullWarp  = (warpIdx + 1) * WARP <= state.bx * state.by;
      const alpha = fullWarp ? 0.9 : 0.4;
      const cx = px + i * tw + 1;
      const cy = py + j * th + 1;
      s += `<rect x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${(tw - 2).toFixed(1)}" height="${(th - 2).toFixed(1)}" fill="rgba(79,106,173,${alpha})"/>`;
    }
  }
  // warp grouping outlines (every WARP threads in row-major)
  const totalThreads = state.bx * state.by;
  for (let w = 0; w < Math.ceil(totalThreads / WARP); w++) {
    const start = w * WARP;
    const end   = Math.min(totalThreads - 1, start + WARP - 1);
    const sxr = (start % state.bx);
    const syr = Math.floor(start / state.bx);
    const exr = (end % state.bx);
    const eyr = Math.floor(end / state.bx);
    if (syr === eyr) {
      // single row warp
      const x1 = px + sxr * tw;
      const x2 = px + (exr + 1) * tw;
      const y1 = py + syr * th;
      const y2 = py + (syr + 1) * th;
      s += `<rect x="${x1.toFixed(1)}" y="${y1.toFixed(1)}" width="${(x2-x1).toFixed(1)}" height="${(y2-y1).toFixed(1)}" fill="none" stroke="#ffffff" stroke-opacity="0.35" stroke-width="0.6"/>`;
    }
  }
  s += `<text x="${px}" y="${py + panelH + 9}" font-family="inherit" font-size="9" fill="#6e7280" letter-spacing="1.4">BLOCK &middot; ${state.bx}&times;${state.by} threads &middot; ${Math.ceil(totalThreads / WARP)} warps</text>`;

  svg.innerHTML = s;
}

function renderResults(r) {
  $('tpbVal').textContent  = r.tpb;
  $('wpbVal').textContent  = r.warpsPerBlock;
  $('gridVal').innerHTML   = r.gridX + ' &times; ' + r.gridY;
  $('ttVal').textContent   = fmtThreads(r.totalThreads);

  $('smUseVal').textContent = `${r.smUsed} / ${state.gpu.sms}`;
  $('bpsVal').textContent   = r.activeBlocksPerSm;

  $('timeVal').textContent  = fmtTime(r.timeSec);
  $('costVal').textContent  = fmtCost(r.costUSD);
  $('scoreVal').textContent = r.score.toFixed(2);

  $('occBar').style.width = pct(r.occupancy);
  $('occVal').textContent = pct(r.occupancy);
  $('weBar').style.width  = pct(r.warpEff);
  $('weVal').textContent  = pct(r.warpEff);
  $('coBar').style.width  = pct(r.coalesce);
  $('coVal').textContent  = pct(r.coalesce);
  $('meBar').style.width  = pct(r.memFac);
  $('meVal').textContent  = pct(r.memFac);
}

function renderAdvice(r) {
  const tips = [];
  if (r.tpb % WARP !== 0) {
    tips.push(`<b>Threads-per-block ${r.tpb}</b> isn't a multiple of 32 &mdash; the last warp runs with idle lanes.`);
  }
  if (state.bx < WARP) {
    tips.push(`<b>Block dim X = ${state.bx}</b> hurts memory coalescing on row-major data &mdash; try 32.`);
  }
  if (r.occupancy < 0.5) {
    tips.push(`<b>Occupancy ${pct(r.occupancy)}</b> is low &mdash; SM slots are going unused. Try a smaller block (more blocks per SM) or fewer threads per block.`);
  }
  if (state.mem === 'global') {
    tips.push(`Re-reading from global memory each iteration is wasteful &mdash; <b>ping-pong</b> or <b>shared-memory tiling</b> will recover bandwidth.`);
  }
  if (state.mem === 'sharedpp' && r.occupancy < 0.4) {
    tips.push(`Shared-memory tiling is paying off on throughput but starving occupancy &mdash; consider a smaller tile.`);
  }
  if (r.tpb > 512 && r.occupancy < 1) {
    tips.push(`Big blocks (${r.tpb} threads) leave fewer slots per SM &mdash; smaller blocks often raise occupancy on this workload.`);
  }
  if (tips.length === 0) {
    tips.push(`<b>Configuration looks healthy.</b> Try a different GPU to see how cost / time trade off.`);
  }
  $('advice').innerHTML = tips.slice(0, 3).join(' &nbsp;&middot;&nbsp; ');
}

function update() {
  const r = simulate(state);
  renderResults(r);
  renderViz(r);
  renderAdvice(r);
  return r;
}

// ============================================================
//  PERSONAL BEST
// ============================================================
function loadBest() {
  try {
    const raw = localStorage.getItem('poissonBest');
    if (raw) return JSON.parse(raw);
  } catch (_) {}
  return null;
}
function saveBest(b) {
  try { localStorage.setItem('poissonBest', JSON.stringify(b)); } catch (_) {}
}
function renderBest() {
  if (!best) {
    $('bestTime').textContent = '—';
    $('bestCost').textContent = '—';
    $('bestConf').textContent = '—';
    return;
  }
  $('bestTime').textContent = fmtTime(best.timeSec);
  $('bestCost').textContent = fmtCost(best.costUSD);
  $('bestConf').innerHTML = `${best.gpu} &middot; ${best.bx}&times;${best.by} &middot; ${MEM_LABEL[best.mem]}`;
}
function maybeUpdateBest(r) {
  // composite score = lower is better
  const composite = r.timeSec * 1000 * (r.costUSD * 1e6);
  const prev = best ? best.timeSec * 1000 * (best.costUSD * 1e6) : Infinity;
  if (composite < prev) {
    best = {
      timeSec: r.timeSec, costUSD: r.costUSD,
      gpu: state.gpu.name, bx: state.bx, by: state.by, mem: state.mem,
    };
    saveBest(best);
    renderBest();
    return true;
  }
  return false;
}

// ============================================================
//  WIRE UP
// ============================================================
function init() {
  renderGpus();

  $('bxSlider').addEventListener('input', (e) => {
    state.bx = +e.target.value;
    $('bxVal').textContent = state.bx;
    update();
  });
  $('bySlider').addEventListener('input', (e) => {
    state.by = +e.target.value;
    $('byVal').textContent = state.by;
    update();
  });
  document.querySelectorAll('input[name="mem"]').forEach(r => {
    r.addEventListener('change', (e) => {
      state.mem = e.target.value;
      update();
    });
  });

  $('runBtn').addEventListener('click', () => {
    const r = update();
    const improved = maybeUpdateBest(r);
    const btn = $('runBtn');
    btn.textContent = improved ? 'new best ✓' : 'simulated ✓';
    btn.classList.add('flash');
    setTimeout(() => {
      btn.textContent = 'simulate →';
      btn.classList.remove('flash');
    }, 1200);
  });

  $('resetBest').addEventListener('click', () => {
    best = null;
    try { localStorage.removeItem('poissonBest'); } catch (_) {}
    renderBest();
  });

  renderBest();
  update();
}

document.addEventListener('DOMContentLoaded', init);
