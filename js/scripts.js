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
});
