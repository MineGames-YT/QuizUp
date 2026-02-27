// кастомный выпадающий список для select
// - не зависит от библиотек
// - меню рендерится в body (не обрезается overflow у карточек)

(function () {
  'use strict';

  const ACTIVE_CLASS = 'qselect-open';
  let opened = null;

  function isHiddenOption(opt) {
    return opt.disabled && !opt.value;
  }

  function closeOpened() {
    if (!opened) return;
    opened.root.classList.remove(ACTIVE_CLASS);
    opened.menu.classList.remove('show');
    opened.menu.hidden = true;
    opened.trigger.setAttribute('aria-expanded', 'false');
    opened = null;
  }

  function placeMenu(instance) {
    const r = instance.trigger.getBoundingClientRect();
    const menu = instance.menu;

    menu.style.width = r.width + 'px';
    menu.hidden = false;

    const gap = 8;
    const viewportH = window.innerHeight || document.documentElement.clientHeight;
    const menuH = Math.min(menu.scrollHeight, 280);
    const spaceBelow = viewportH - r.bottom;
    const openUp = spaceBelow < menuH + gap && r.top > menuH + gap;

    const top = openUp ? (r.top - menuH - gap) : (r.bottom + gap);

    menu.style.left = Math.round(r.left) + 'px';
    menu.style.top = Math.round(top) + 'px';
    menu.style.width = Math.round(r.width) + 'px';
    menu.style.maxHeight = Math.round(menuH) + 'px';
  }

  function setValue(instance, value) {
    const select = instance.select;
    select.value = value;
    select.dispatchEvent(new Event('change', { bubbles: true }));

    const opt = select.selectedOptions && select.selectedOptions[0];
    instance.valueEl.textContent = opt ? opt.textContent.trim() : instance.placeholder;

    instance.menu.querySelectorAll('.qselect-option').forEach(el => {
      el.classList.toggle('selected', el.getAttribute('data-value') === value);
      el.setAttribute('aria-selected', el.classList.contains('selected') ? 'true' : 'false');
    });
  }

  function open(instance) {
    if (opened && opened !== instance) closeOpened();
    opened = instance;
    instance.root.classList.add(ACTIVE_CLASS);
    instance.trigger.setAttribute('aria-expanded', 'true');
    placeMenu(instance);
    requestAnimationFrame(() => instance.menu.classList.add('show'));
  }

  function buildMenu(instance) {
    const select = instance.select;
    const menu = document.createElement('div');
    menu.className = 'qselect-menu';
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;

    const list = document.createElement('div');
    list.className = 'qselect-list';

    Array.from(select.options).forEach(opt => {
      if (isHiddenOption(opt)) return;

      const item = document.createElement('div');
      item.className = 'qselect-option';
      item.setAttribute('role', 'option');
      item.setAttribute('data-value', opt.value);
      item.textContent = opt.textContent;
      if (opt.disabled) item.classList.add('disabled');

      if (opt.value === select.value) {
        item.classList.add('selected');
        item.setAttribute('aria-selected', 'true');
      } else {
        item.setAttribute('aria-selected', 'false');
      }

      item.addEventListener('click', (e) => {
        e.preventDefault();
        if (item.classList.contains('disabled')) return;
        setValue(instance, opt.value);
        closeOpened();
      });

      list.appendChild(item);
    });

    menu.appendChild(list);
    document.body.appendChild(menu);
    instance.menu = menu;
  }

  function initSelect(select) {
    if (!select || select.__qselect) return;
    select.__qselect = true;

    const placeholderOpt = Array.from(select.options).find(o => isHiddenOption(o));
    const placeholder = placeholderOpt ? placeholderOpt.textContent.trim() : (select.getAttribute('data-placeholder') || 'выбрать');

    // прячем нативный select, но оставляем в dom
    select.classList.add('qselect-native');

    const root = document.createElement('div');
    root.className = 'qselect';
    root.setAttribute('data-select-id', select.id || '');

    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'qselect-trigger';
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');

    const valueEl = document.createElement('span');
    valueEl.className = 'qselect-value';

    const currentOpt = select.selectedOptions && select.selectedOptions[0];
    valueEl.textContent = currentOpt ? currentOpt.textContent.trim() : placeholder;

    const chevron = document.createElement('span');
    chevron.className = 'qselect-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    chevron.textContent = '▾';

    trigger.appendChild(valueEl);
    trigger.appendChild(chevron);
    root.appendChild(trigger);

    // вставляем root рядом с select
    select.insertAdjacentElement('afterend', root);

    const instance = {
      select,
      root,
      trigger,
      valueEl,
      placeholder,
      menu: null
    };

    buildMenu(instance);

    trigger.addEventListener('click', (e) => {
      e.preventDefault();
      if (opened && opened === instance) {
        closeOpened();
      } else {
        open(instance);
      }
    });

    trigger.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeOpened();
      }
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        trigger.click();
      }
    });

    // если select меняется программно — синхронизируем текст
    select.addEventListener('change', () => {
      const opt = select.selectedOptions && select.selectedOptions[0];
      valueEl.textContent = opt ? opt.textContent.trim() : placeholder;
      instance.menu.querySelectorAll('.qselect-option').forEach(el => {
        el.classList.toggle('selected', el.getAttribute('data-value') === select.value);
      });
    });
  }

  function initAll() {
    document.querySelectorAll('select.js-custom-select').forEach(initSelect);

    document.addEventListener('click', (e) => {
      if (!opened) return;
      const withinTrigger = opened.root.contains(e.target);
      const withinMenu = opened.menu && opened.menu.contains(e.target);
      if (!withinTrigger && !withinMenu) closeOpened();
    });

    window.addEventListener('resize', () => {
      if (opened) placeMenu(opened);
    });

    window.addEventListener('scroll', () => {
      if (opened) placeMenu(opened);
    }, true);
  }

  document.addEventListener('DOMContentLoaded', initAll);
})();
