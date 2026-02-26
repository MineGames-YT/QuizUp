(function () {
  'use strict';

  function esc(s) {
    try {
      if (window.CSS && typeof CSS.escape === 'function') return CSS.escape(s);
    } catch (e) {}
    return String(s).replace(/"/g, '\\"');
  }

  function clampText(s, maxLen) {
    if (!s) return '';
    s = String(s).trim();
    if (s.length <= maxLen) return s;
    return s.slice(0, maxLen);
  }

  // ---------------- theme ----------------
  function applyTheme(theme) {
    const html = document.documentElement;
    html.setAttribute('data-theme', theme);
    const btn = document.getElementById('themeToggle');
    if (btn) {
      btn.setAttribute('aria-label', theme === 'dark' ? 'включить светлую тему' : 'включить тёмную тему');
      btn.setAttribute('title', theme === 'dark' ? 'включить светлую тему' : 'включить тёмную тему');
    }
  }

  function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    applyTheme(saved === 'dark' ? 'dark' : 'light');

    const btn = document.getElementById('themeToggle');
    if (btn) {
      btn.addEventListener('click', function () {
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        const next = (current === 'dark') ? 'light' : 'dark';
        localStorage.setItem('theme', next);
        applyTheme(next);
      });
    }
  }

  // ---------------- toast notifications ----------------
  function ensureToastContainer() {
    let c = document.getElementById('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      c.className = 'toast-container';
      document.body.appendChild(c);
    }
    return c;
  }

  function toast(message, type, timeoutMs) {
    message = (message == null) ? '' : String(message);
    type = type || 'info';
    timeoutMs = Number.isFinite(timeoutMs) ? timeoutMs : 3200;

    const container = ensureToastContainer();
    const el = document.createElement('div');
    el.className = 'toast toast-' + type;
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');

    const text = document.createElement('div');
    text.className = 'toast-text';
    text.textContent = message;

    const close = document.createElement('button');
    close.className = 'toast-close';
    close.type = 'button';
    close.setAttribute('aria-label', 'закрыть');
    close.innerHTML = '<span aria-hidden="true">×</span>';

    close.addEventListener('click', function () {
      removeToast(el);
    });

    el.appendChild(text);
    el.appendChild(close);
    container.appendChild(el);

    // force animation
    requestAnimationFrame(function () {
      el.classList.add('show');
    });

    if (timeoutMs > 0) {
      setTimeout(function () {
        removeToast(el);
      }, timeoutMs);
    }
  }

  function removeToast(el) {
    if (!el || el.__removing) return;
    el.__removing = true;
    el.classList.remove('show');
    el.classList.add('hide');
    setTimeout(function () {
      try { el.remove(); } catch (e) {}
    }, 220);
  }

  // ---------------- confirm dialog ----------------
  function ensureConfirm() {
    let overlay = document.getElementById('qbConfirm');
    if (overlay) return overlay;

    overlay = document.createElement('div');
    overlay.id = 'qbConfirm';
    overlay.className = 'confirm-overlay';
    overlay.hidden = true;

    overlay.innerHTML = [
      '<div class="confirm-card" role="dialog" aria-modal="true" aria-labelledby="qbConfirmTitle">',
      '  <div class="confirm-title" id="qbConfirmTitle">подтвердить</div>',
      '  <div class="confirm-message" id="qbConfirmMessage"></div>',
      '  <div class="confirm-actions">',
      '    <button class="btn btn-secondary" type="button" id="qbConfirmCancel">отмена</button>',
      '    <button class="btn btn-danger" type="button" id="qbConfirmOk">да</button>',
      '  </div>',
      '</div>'
    ].join('\n');

    document.body.appendChild(overlay);
    return overlay;
  }

  function confirmDialog(opts) {
    opts = opts || {};
    const overlay = ensureConfirm();
    const titleEl = overlay.querySelector('#qbConfirmTitle');
    const msgEl = overlay.querySelector('#qbConfirmMessage');
    const okBtn = overlay.querySelector('#qbConfirmOk');
    const cancelBtn = overlay.querySelector('#qbConfirmCancel');

    titleEl.textContent = opts.title || 'подтвердить';
    msgEl.textContent = opts.message || '';
    okBtn.textContent = opts.okText || 'да';
    cancelBtn.textContent = opts.cancelText || 'отмена';

    overlay.hidden = false;
    overlay.classList.add('show');

    return new Promise(function (resolve) {
      let done = false;

      function cleanup(result) {
        if (done) return;
        done = true;
        overlay.classList.remove('show');
        overlay.classList.add('hide');
        setTimeout(function () {
          overlay.hidden = true;
          overlay.classList.remove('hide');
        }, 200);
        document.removeEventListener('keydown', onKey);
        overlay.removeEventListener('click', onOverlayClick);
        okBtn.removeEventListener('click', onOk);
        cancelBtn.removeEventListener('click', onCancel);
        resolve(result);
      }

      function onOk() { cleanup(true); }
      function onCancel() { cleanup(false); }
      function onOverlayClick(e) {
        if (e.target === overlay) cleanup(false);
      }
      function onKey(e) {
        if (e.key === 'Escape') cleanup(false);
      }

      okBtn.addEventListener('click', onOk);
      cancelBtn.addEventListener('click', onCancel);
      overlay.addEventListener('click', onOverlayClick);
      document.addEventListener('keydown', onKey);

      // focus
      setTimeout(function () {
        okBtn.focus();
      }, 0);
    });
  }

  // ---------------- custom select ----------------
  function closeAllSelects(except) {
    document.querySelectorAll('.cselect.open').forEach(function (el) {
      if (except && el === except) return;
      el.classList.remove('open');
      const trigger = el.querySelector('.cselect-trigger');
      if (trigger) trigger.setAttribute('aria-expanded', 'false');
    });
  }

  function setSelectValue(root, value, labelText) {
    const hidden = root.querySelector('input[type="hidden"]');
    const valueEl = root.querySelector('.cselect-value');

    if (hidden) hidden.value = value;
    if (valueEl) valueEl.textContent = labelText || '';

    // mark selected option
    root.querySelectorAll('.cselect-option').forEach(function (opt) {
      opt.classList.toggle('selected', opt.getAttribute('data-value') === value);
    });

    root.dispatchEvent(new CustomEvent('cselect:change', { detail: { value: value } }));
  }

  function initOneSelect(root) {
    const trigger = root.querySelector('.cselect-trigger');
    const menu = root.querySelector('.cselect-menu');
    const hidden = root.querySelector('input[type="hidden"]');
    const placeholder = root.getAttribute('data-placeholder') || 'выбрать';
    const valueEl = root.querySelector('.cselect-value');

    if (!trigger || !menu || !hidden || !valueEl) return;

    // initial
    const initialValue = hidden.value || '';
    const selectedOpt = root.querySelector('.cselect-option[data-value="' + esc(initialValue) + '"]');
    if (selectedOpt) {
      setSelectValue(root, initialValue, selectedOpt.textContent.trim());
    } else {
      valueEl.textContent = placeholder;
    }

    trigger.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      const isOpen = root.classList.contains('open');
      closeAllSelects();
      root.classList.toggle('open', !isOpen);
      trigger.setAttribute('aria-expanded', (!isOpen).toString());
    });

    menu.addEventListener('click', function (e) {
      const opt = e.target.closest('.cselect-option');
      if (!opt) return;
      const val = opt.getAttribute('data-value') || '';
      setSelectValue(root, val, opt.textContent.trim());
      closeAllSelects();
    });

    // keyboard support (basic)
    trigger.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        trigger.click();
      }
      if (e.key === 'Escape') {
        closeAllSelects();
      }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!root.classList.contains('open')) {
          trigger.click();
          return;
        }
        const options = Array.from(root.querySelectorAll('.cselect-option'));
        const current = options.findIndex(o => o.classList.contains('selected'));
        const next = (e.key === 'ArrowDown')
          ? Math.min(options.length - 1, current + 1)
          : Math.max(0, current - 1);
        const opt = options[next];
        if (opt) {
          options.forEach(o => o.classList.remove('focused'));
          opt.classList.add('focused');
          opt.scrollIntoView({ block: 'nearest' });
        }
      }
    });

    menu.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeAllSelects();
        trigger.focus();
      }
      if (e.key === 'Enter') {
        const focused = root.querySelector('.cselect-option.focused') || root.querySelector('.cselect-option.selected');
        if (focused) {
          setSelectValue(root, focused.getAttribute('data-value') || '', focused.textContent.trim());
          closeAllSelects();
          trigger.focus();
        }
      }
    });

    // focus styling
    root.addEventListener('focusin', function () {
      root.classList.add('focus');
    });
    root.addEventListener('focusout', function () {
      root.classList.remove('focus');
    });
  }

  function initSelects() {
    document.querySelectorAll('.cselect').forEach(initOneSelect);

    document.addEventListener('click', function () {
      closeAllSelects();
    });

    window.addEventListener('blur', function () {
      closeAllSelects();
    });
  }

  // expose helper for name storage
  function getStoredPlayerName() {
    const v = (localStorage.getItem('qb_player_name') || '').trim();
    if (v) return v;
    // backward compatibility
    return (localStorage.getItem('qb_guest_name') || '').trim();
  }

  function setStoredPlayerName(v) {
    v = clampText(v, 20);
    localStorage.setItem('qb_player_name', v);
    // backward compatibility for older pages
    localStorage.setItem('qb_guest_name', v);
  }

  window.QB_UI = {
    initTheme,
    initSelects,
    getStoredPlayerName,
    setStoredPlayerName,
    toast,
    confirm: confirmDialog
  };

  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initSelects();

    // flash messages from server
    try {
      const msgs = window.__flash_messages;
      if (Array.isArray(msgs) && msgs.length) {
        msgs.forEach(m => toast(m, 'error', 4500));
        window.__flash_messages = [];
      }
    } catch (e) {}
  });
})();
