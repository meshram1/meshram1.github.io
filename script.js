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
