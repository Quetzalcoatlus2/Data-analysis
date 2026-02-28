(function () {
      const saved = localStorage.getItem('theme') || 'light';
      document.documentElement.setAttribute('data-theme', saved);
      function setThemeButtonLabel() {
        const btn = document.getElementById('themeToggle');
        if (!btn) return;
        const cur = document.documentElement.getAttribute('data-theme') || 'light';
        btn.textContent = cur === 'light' ? 'Dark mode' : 'Light mode';
      }
      window.toggleTheme = function () {
        const cur = document.documentElement.getAttribute('data-theme') || 'light';
        const next = cur === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        if (window.applyThemeToPlots) window.applyThemeToPlots();
        setThemeButtonLabel();
      };
      window.addEventListener('DOMContentLoaded', setThemeButtonLabel);
      window.getThemeVars = function () {
        const s = getComputedStyle(document.documentElement);
        const pick = n => (s.getPropertyValue(n) || '').trim();
        return {
          bg: pick('--bg'),
          panel: pick('--panel'),
          text: pick('--text'),
          border: pick('--border'),
          accent: pick('--accent'),
          accent2: pick('--accent-2')
        };
      };
      // Helper to detect dark mode
      window.isDarkMode = function () {
        const theme = document.documentElement.getAttribute('data-theme');
        if (theme) return theme === 'dark';
        // Fallback: check background color
        const t = getThemeVars();
        return t.bg && /^#0|^#1|^#2/i.test(t.bg.trim());
      };

      // Global theme callback registry - views can register their update functions
      window.themeCallbacks = window.themeCallbacks || [];

      // Master function to apply theme to all registered plots
      window.applyThemeToPlots = function () {
        // Call all registered theme callbacks
        if (window.themeCallbacks && window.themeCallbacks.length > 0) {
          window.themeCallbacks.forEach(function (callback) {
            if (typeof callback === 'function') {
              try { callback(); } catch (e) { console.error('Theme callback error:', e); }
            }
          });
        }
      };
    })();

(function () {
      let p;
      window.loadPlotly = function loadPlotly() {
        if (window.Plotly) return Promise.resolve(window.Plotly);
        if (p) return p;
        p = new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
          s.async = true;
          s.onload = () => resolve(window.Plotly);
          s.onerror = (e) => reject(new Error('Failed to load Plotly'));
          document.head.appendChild(s);
        });
        return p;
      };
    })();

(function () {
          const inp = document.getElementById('questionInput');
          const btn = document.getElementById('askBtn');
          const form = document.getElementById('qaForm');
          function sync() {
            const v = (inp.value || '').trim();
            btn.disabled = v.length === 0;
          }
          if (inp && btn) {
            inp.addEventListener('input', sync);
            sync();
            form.addEventListener('submit', function (e) {
              if (btn.disabled) e.preventDefault();
              else sessionStorage.setItem('scrollPos', window.scrollY.toString());
            });
          }
          // Only restore scroll position for normal page reloads (not Q&A submissions)
          // The actual scroll-to-answer happens in a script AFTER the Q&A elements are rendered
          const savedScroll = sessionStorage.getItem('scrollPos');
          if (savedScroll && !document.getElementById('ai-answer-display')) {
            window.scrollTo(0, parseInt(savedScroll, 10));
            sessionStorage.removeItem('scrollPos');
          }
        })();

(function () {
          const questionDisplay = document.getElementById('ai-question-display');
          if (questionDisplay) {
            // Scroll the QUESTION to the CENTER of viewport
            // This positions the question in the middle so the answer is visible below
            setTimeout(() => {
              questionDisplay.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 150);
            sessionStorage.removeItem('scrollPos');
          }
        })();

