document.addEventListener('DOMContentLoaded', () => {
  /* ---------------- theme toggle (dark <-> light) ---------------- */
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');

  const applyTheme = (theme, persist) => {
    if (theme === 'light') root.setAttribute('data-theme', 'light');
    else root.removeAttribute('data-theme');
    if (toggle) toggle.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
    if (themeColorMeta) themeColorMeta.setAttribute('content', theme === 'light' ? '#f7f7f8' : '#0a0a0a');
    if (persist) { try { localStorage.setItem('theme', theme); } catch (e) { /* ignore */ } }
  };

  let currentTheme = root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  applyTheme(currentTheme, false);

  if (toggle) {
    toggle.addEventListener('click', () => {
      currentTheme = currentTheme === 'light' ? 'dark' : 'light';
      applyTheme(currentTheme, true);
    });
  }

  /* ---------------- play videos only while in view ---------------- */
  const videos = Array.from(document.querySelectorAll('video[data-autoplay]'));
  if ('IntersectionObserver' in window && videos.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const v = entry.target;
        if (entry.isIntersecting) {
          const p = v.play();
          if (p && p.catch) p.catch(() => { /* autoplay may be blocked; ignore */ });
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.2 });
    videos.forEach((v) => io.observe(v));
  } else {
    videos.forEach((v) => { const p = v.play(); if (p && p.catch) p.catch(() => {}); });
  }

  /* ---------------- A/B comparison sliders ---------------- */
  document.querySelectorAll('.ab-slider').forEach((slider) => {
    const top = slider.querySelector('.ab-top');
    const handle = slider.querySelector('.ab-handle');
    const inner = slider.querySelector('.ab-top .ab-inner');
    const vids = Array.from(slider.querySelectorAll('video'));

    const setPos = (ratio) => {
      const r = Math.max(0, Math.min(1, ratio));
      const pct = (r * 100).toFixed(2) + '%';
      top.style.width = pct;
      handle.style.left = pct;
      // counter-scale the clipped (top) video so both halves stay pixel-aligned
      if (inner) inner.style.width = slider.clientWidth + 'px';
    };

    const ratioFromEvent = (e) => {
      const rect = slider.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      return x / rect.width;
    };

    let dragging = false;
    const start = (e) => { dragging = true; setPos(ratioFromEvent(e)); e.preventDefault(); };
    const move = (e) => { if (dragging) setPos(ratioFromEvent(e)); };
    const end = () => { dragging = false; };

    slider.addEventListener('pointerdown', start);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', end);
    // touch fallback for browsers without pointer events
    slider.addEventListener('touchstart', start, { passive: false });
    window.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('touchend', end);

    // keep the two clips time-synced; re-measure on resize
    const sync = () => { if (vids.length === 2 && Math.abs(vids[0].currentTime - vids[1].currentTime) > 0.08) vids[1].currentTime = vids[0].currentTime; };
    if (vids[0]) vids[0].addEventListener('timeupdate', sync);
    window.addEventListener('resize', () => setPos(parseFloat(top.style.width) / 100 || 0.5));

    setPos(0.5);
  });

  /* ---------------- BibTeX copy ---------------- */
  document.querySelectorAll('[data-copy-target]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const pre = document.querySelector(btn.getAttribute('data-copy-target'));
      if (!pre) return;
      const text = pre.innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const r = document.createRange(); r.selectNode(pre);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        try { document.execCommand('copy'); } catch (_) {}
        s.removeAllRanges();
      }
      const original = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = original; btn.classList.remove('copied'); }, 1600);
    });
  });

  /* ---------------- results explorer (Ours vs. baseline) ---------------- */
  (function initExplorer() {
    if (typeof GALLERY === 'undefined') return;
    const root = document.getElementById('explorer');
    if (!root) return;

    const slider = document.getElementById('exp-slider');
    const vidOurs = document.getElementById('exp-vid-ours');
    const vidBase = document.getElementById('exp-vid-base');
    const labOurs = document.getElementById('exp-label-ours');
    const labBase = document.getElementById('exp-label-base');
    const todo = document.getElementById('exp-todo');
    const caption = document.getElementById('exp-caption');

    const groups = {};
    root.querySelectorAll('.exp-group').forEach((g) => { groups[g.dataset.axis] = g.querySelector('.exp-btns'); });

    const dsList = Object.keys(GALLERY.datasets);
    const state = { ds: 'gi', ours: 'GraphiTS', baseline: 'Ex4DGS', scene: 'flames', cond: 'faulty_cam_2' };

    let haveSet = new Set();
    const curDS = () => GALLERY.datasets[state.ds];
    const has = (m, c, s) => haveSet.has(m + '/' + c + '/' + s);
    const find = (arr, id) => arr.find((x) => x.id === id) || null;
    const methodLabel = (id) => (find(curDS().methods, id) || {}).label || id;
    const sceneLabel = (id) => (find(curDS().scenes, id) || {}).label || id;
    const condOf = (id) => find(curDS().conditions, id) || { label: id, group: '' };

    function makeChip(label, opts) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'exp-chip' + (opts.cls ? ' ' + opts.cls : '');
      b.textContent = label;
      if (opts.active) b.classList.add('active');
      if (opts.todo) {
        b.classList.add('todo');
        const t = document.createElement('span');
        t.className = 'tag';
        t.textContent = 'todo';
        b.appendChild(t);
      }
      b.addEventListener('click', opts.onClick);
      return b;
    }

    function render() {
      const ds = curDS();
      const ours = ds.methods.filter((m) => m.ours);
      const baselines = ds.methods.filter((m) => !m.ours);

      groups.ds.innerHTML = '';
      dsList.forEach((id) => groups.ds.appendChild(makeChip(GALLERY.datasets[id].label, {
        active: state.ds === id,
        onClick: () => switchDataset(id),
      })));

      groups.ours.innerHTML = '';
      ours.forEach((m) => groups.ours.appendChild(makeChip(m.label, {
        cls: 'ours',
        active: state.ours === m.id,
        todo: !has(m.id, state.cond, state.scene),
        onClick: () => { state.ours = m.id; update(); },
      })));

      groups.baseline.innerHTML = '';
      baselines.forEach((m) => groups.baseline.appendChild(makeChip(m.label, {
        cls: m.gt ? 'gt' : '',
        active: state.baseline === m.id,
        todo: !has(m.id, state.cond, state.scene),
        onClick: () => { state.baseline = m.id; update(); },
      })));

      groups.scene.innerHTML = '';
      ds.scenes.forEach((s) => groups.scene.appendChild(makeChip(s.label, {
        active: state.scene === s.id,
        todo: !has(state.ours, state.cond, s.id),
        onClick: () => { state.scene = s.id; update(); },
      })));

      // Setting: grouped under sub-labels (Standard / Sparse Views / ...)
      groups.cond.innerHTML = '';
      let curGroup = null;
      let sub = null;
      ds.conditions.forEach((c) => {
        if (c.group !== curGroup) {
          curGroup = c.group;
          sub = document.createElement('div');
          sub.className = 'exp-subgroup';
          const lab = document.createElement('span');
          lab.className = 'exp-sublabel';
          lab.textContent = c.group;
          sub.appendChild(lab);
          groups.cond.appendChild(sub);
        }
        sub.appendChild(makeChip(c.label, {
          active: state.cond === c.id,
          todo: !has(state.ours, c.id, state.scene),
          onClick: () => { state.cond = c.id; update(); },
        }));
      });
    }

    const srcFor = (m) => GALLERY.base + state.ds + '/' + m + '/' + state.cond + '/' + state.scene + '.mp4';
    function applyVideo(v, m, playing) {
      const src = srcFor(m);
      if (v.getAttribute('src') !== src) { v.setAttribute('src', src); v.load(); }
      if (playing) { const p = v.play(); if (p && p.catch) p.catch(() => {}); } else { v.pause(); }
    }
    function clearVideo(v) { v.pause(); if (v.getAttribute('src')) { v.removeAttribute('src'); v.load(); } }

    function update() {
      render();
      const okOurs = has(state.ours, state.cond, state.scene);
      const okBase = has(state.baseline, state.cond, state.scene);

      labOurs.innerHTML = '<span class="ours">' + methodLabel(state.ours) + '</span> (Ours)';
      labBase.textContent = methodLabel(state.baseline);

      const c = condOf(state.cond);
      const gw = (c.group || '').split(' ')[0];
      const condText = (!c.group || c.label === c.group || c.label.indexOf(gw) === 0)
        ? c.label : c.group + ' (' + c.label + ')';
      caption.innerHTML = '<span class="ours">' + methodLabel(state.ours) + '</span> vs. ' +
        methodLabel(state.baseline) + ' &middot; ' + sceneLabel(state.scene) + ' &middot; ' +
        condText + ' &middot; ' + curDS().label;

      if (okOurs && okBase) {
        todo.hidden = true;
        slider.style.visibility = '';
        applyVideo(vidBase, state.baseline, true);
        applyVideo(vidOurs, state.ours, true);
      } else {
        slider.style.visibility = 'hidden';
        const missing = [];
        if (okOurs) applyVideo(vidOurs, state.ours, false);
        else { missing.push(methodLabel(state.ours) + ' (Ours)'); clearVideo(vidOurs); }
        if (okBase) applyVideo(vidBase, state.baseline, false);
        else { missing.push(methodLabel(state.baseline)); clearVideo(vidBase); }
        todo.innerHTML = '<div><b>TODO</b> &mdash; ' + missing.join(' and ') +
          ' not yet rendered<br>for ' + sceneLabel(state.scene) + ' &middot; ' + condText + '.</div>';
        todo.hidden = false;
      }
    }

    function switchDataset(id) {
      if (id === state.ds) { return; }
      state.ds = id;
      haveSet = new Set(curDS().have);
      const ds = curDS();
      state.scene = ds.scenes[0].id;
      state.cond = 'standard';
      if (!ds.methods.some((m) => m.id === state.baseline && !m.ours)) state.baseline = 'FTGS';
      if (!ds.methods.some((m) => m.id === state.ours && m.ours)) state.ours = 'GraphiTS';
      update();
    }

    haveSet = new Set(curDS().have);
    update();
  })();
});
