// ============================================================
//  add FNO data-center cooling project
// ============================================================
const projectList = document.getElementById('projectList');

if (projectList && !projectList.querySelector('[data-project="fno-datacenter-cooling"]')) {
  const fnoProject = document.createElement('article');
  fnoProject.className = 'p-item';
  fnoProject.dataset.project = 'fno-datacenter-cooling';
  fnoProject.dataset.tags = 'ml cfd python gpu cuda';
  fnoProject.innerHTML = `
    <div class="p-meta">
      <span class="p-no"></span>
      <span class="p-date">2026</span>
    </div>
    <div class="p-body">
      <h3 class="p-title">Fourier Neural Operator for Data-Center Cooling Design</h3>
      <p class="p-sub">Differentiable CFD surrogate for automated cooling-layout optimization</p>
      <p class="p-by">
        <span>built with</span>
        <span class="p-by-link">PyTorch &middot; CUDA &middot; FNO</span>
        <span>&middot;</span>
        <a href="https://github.com/meshram1/FNO-Datacenter-Cooling" target="_blank" rel="noopener" class="p-by-link">code &rarr;</a>
      </p>
      <div class="p-tags">
        <span class="tag">ml</span>
        <span class="tag">cfd</span>
        <span class="tag">python</span>
        <span class="tag">gpu</span>
        <span class="tag">cuda</span>
      </div>
      <p class="p-desc">
        Built an end-to-end <strong>physics-ML pipeline</strong> for data-center
        cooling: a GPU pseudo-spectral CFD solver generates buoyancy-driven
        airflow data, then a <strong>Fourier Neural Operator</strong> learns the
        solution operator for fast, differentiable prediction. Backpropagating
        through the surrogate enables <strong>gradient-based cooling-layout
        optimization</strong>, reducing peak temperature by
        <strong>11.5%</strong> through rack placement and
        <strong>15.7%</strong> through cold-vent placement with fixed racks.
      </p>
    </div>
  `;

  projectList.prepend(fnoProject);

  // Keep project numbering consistent with the existing reverse-chronological list.
  projectList.querySelectorAll('.p-item').forEach((item, index) => {
    const number = item.querySelector('.p-no');
    if (number) number.textContent = String(index + 1).padStart(2, '0');
  });
}

// ============================================================
//  project dates — show completion year only
// ============================================================
document.querySelectorAll('.p-date').forEach((date) => {
  const match = date.textContent.match(/(20\d{2})/);
  if (match) date.textContent = match[1];
});

// ============================================================
//  reveal on scroll
// ============================================================
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    });
  },
  { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
);
document
  .querySelectorAll('.p-item, .exp-item, .edu-item, .skill-block, .link-row')
  .forEach((el) => io.observe(el));

// ============================================================
//  filter pills — multi-select (AND across selected tags)
// ============================================================
const pillsEl = document.getElementById('pills');
const itemsEl = document.getElementById('projectList');
const emptyNote = document.getElementById('emptyNote');

if (pillsEl && itemsEl) {
  const pills = Array.from(pillsEl.querySelectorAll('.pill'));
  const items = Array.from(itemsEl.querySelectorAll('.p-item'));

  function activeFilters() {
    return pills
      .filter((p) => p.classList.contains('active') && p.dataset.filter !== 'all')
      .map((p) => p.dataset.filter);
  }

  function applyFilters() {
    const filters = activeFilters();
    let visibleCount = 0;

    items.forEach((it) => {
      const tags = (it.dataset.tags || '').split(/\s+/);
      const match = filters.length === 0 || filters.every((f) => tags.includes(f));
      it.classList.toggle('hidden', !match);
      if (match) visibleCount++;
    });

    if (emptyNote) emptyNote.hidden = visibleCount !== 0;

    const allPill = pills.find((p) => p.dataset.filter === 'all');
    if (allPill) {
      allPill.classList.toggle('active', filters.length === 0);
      allPill.setAttribute('aria-pressed', filters.length === 0 ? 'true' : 'false');
    }
  }

  pills.forEach((p) => {
    p.addEventListener('click', () => {
      if (p.dataset.filter === 'all') {
        pills.forEach((q) => q.classList.remove('active'));
        p.classList.add('active');
      } else {
        p.classList.toggle('active');
      }
      applyFilters();
    });
  });

  applyFilters();
}

// ============================================================
//  nav active state on scroll
// ============================================================
const navLinks = document.querySelectorAll('.nav-links a[data-nav]');
const sectionMap = new Map();
navLinks.forEach((a) => {
  const id = a.getAttribute('href').slice(1);
  const sec = document.getElementById(id);
  if (sec) sectionMap.set(sec, a);
});

const navIO = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        navLinks.forEach((a) => a.classList.remove('active'));
        const link = sectionMap.get(e.target);
        if (link) link.classList.add('active');
      }
    });
  },
  { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
);
sectionMap.forEach((_, sec) => navIO.observe(sec));
