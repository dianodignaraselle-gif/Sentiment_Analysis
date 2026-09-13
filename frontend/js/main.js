// Pages that opt in with <body data-auth="required"> are only reachable
// while logged in; anyone else gets bounced to the login page immediately.
(function enforceAuth() {
  if (document.body.dataset.auth === "required" && !localStorage.getItem("displayName")) {
    const next = window.location.pathname.split("/").pop() || "index.html";
    window.location.href = `login.html?next=${encodeURIComponent(next)}`;
  }
})();

function showToast(message, isError = false) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  clearTimeout(toast._hideTimer);
  toast._hideTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function sentimentEmoji(sentiment) {
  const emoji = { positive: "😊", neutral: "😐", negative: "😞" }[sentiment] || "😐";
  return `<span class="emoji-dot ${sentiment}">${emoji}</span>`;
}

function initThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    // SVG charts (charts.js) are drawn once with colors read from CSS
    // variables at render time; reload so any already-drawn chart repaints
    // in the new theme instead of staying stuck with stale colors.
    if (document.querySelector("#dist-donut, #time-chart, #product-chart, #overall-donut")) {
      window.location.reload();
    }
  });
}

// Only "displayName" ever represents an authenticated session in this app
// (see api.js currentOwner()) - clearing it is a full, secure logout.
function clearAuthSession() {
  localStorage.removeItem("displayName");
}

function initLoginButton() {
  const btn = document.getElementById("login-btn");
  const profileMenu = document.getElementById("profile-menu");
  const savedName = localStorage.getItem("displayName");

  if (btn) {
    if (savedName) {
      btn.hidden = true;
    } else {
      btn.addEventListener("click", () => {
        const next = window.location.pathname.split("/").pop() || "index.html";
        window.location.href = `login.html?next=${encodeURIComponent(next)}`;
      });
    }
  }

  if (profileMenu && savedName) {
    profileMenu.hidden = false;
    const avatar = document.getElementById("profile-avatar");
    const name = document.getElementById("profile-name");
    // Only the first letter of the entered email/username is ever shown -
    // the full value stays out of the DOM entirely for privacy.
    if (avatar) avatar.textContent = savedName.trim().charAt(0).toUpperCase() || "?";
    if (name) name.hidden = true;
    // profile-name (the only visible label for this button) is now hidden,
    // so the trigger needs its own accessible name that still avoids
    // exposing the full email/username.
    const trigger = profileMenu.querySelector("#profile-trigger");
    if (trigger) trigger.setAttribute("aria-label", "Account menu");
    initProfileMenu(profileMenu);
  }
}

// Wires up the avatar/name trigger, its dropdown (open/close, outside click,
// Esc, roving focus), and hands off to the logout confirmation modal.
function initProfileMenu(container) {
  const trigger = container.querySelector("#profile-trigger");
  const dropdown = container.querySelector("#profile-dropdown");
  const logoutTrigger = container.querySelector("#logout-trigger");
  if (!trigger || !dropdown || !logoutTrigger) return;

  let hideTimer = null;

  const onOutsideClick = (e) => {
    if (!container.contains(e.target)) closeDropdown();
  };

  const onKeydown = (e) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      closeDropdown(true);
    }
  };

  function openDropdown() {
    clearTimeout(hideTimer);
    dropdown.hidden = false;
    requestAnimationFrame(() => dropdown.classList.add("open"));
    trigger.setAttribute("aria-expanded", "true");
    document.addEventListener("click", onOutsideClick);
    document.addEventListener("keydown", onKeydown);
  }

  function closeDropdown(restoreFocus = false) {
    dropdown.classList.remove("open");
    trigger.setAttribute("aria-expanded", "false");
    document.removeEventListener("click", onOutsideClick);
    document.removeEventListener("keydown", onKeydown);
    clearTimeout(hideTimer);
    // Keep the dropdown in the layout until the fade/slide-out transition
    // (150ms, matches .profile-dropdown in style.css) finishes.
    hideTimer = setTimeout(() => { dropdown.hidden = true; }, 150);
    if (restoreFocus) trigger.focus();
  }

  trigger.addEventListener("click", () => {
    if (trigger.getAttribute("aria-expanded") === "true") {
      closeDropdown();
    } else {
      openDropdown();
      logoutTrigger.focus();
    }
  });

  logoutTrigger.addEventListener("click", () => {
    closeDropdown();
    openLogoutModal(trigger);
  });
}

// The logout confirmation modal is shared across pages and only needed once
// the user actually opens it, so it's built lazily and appended to <body> -
// the same pattern showToast() uses for the toast element.
let logoutModalReturnFocus = null;

function ensureLogoutModal() {
  let overlay = document.getElementById("logout-modal-overlay");
  if (overlay) return overlay;

  overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.id = "logout-modal-overlay";
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="modal-dialog" id="logout-modal" role="alertdialog" aria-modal="true" aria-labelledby="logout-modal-message" tabindex="-1">
      <p class="modal-message" id="logout-modal-message">Are you sure you want to log out?</p>
      <div class="modal-actions">
        <button class="btn btn-outline" id="logout-cancel-btn" type="button">No</button>
        <button class="btn btn-danger" id="logout-confirm-btn" type="button">Log Out</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const cancelBtn = overlay.querySelector("#logout-cancel-btn");
  const confirmBtn = overlay.querySelector("#logout-confirm-btn");
  let hideTimer = null;

  const onKeydown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "Tab") {
      // Two-button focus trap: Tab/Shift+Tab just bounces between them.
      const focusFirst = cancelBtn;
      const focusLast = confirmBtn;
      if (e.shiftKey && document.activeElement === focusFirst) {
        e.preventDefault();
        focusLast.focus();
      } else if (!e.shiftKey && document.activeElement === focusLast) {
        e.preventDefault();
        focusFirst.focus();
      }
    }
  };

  function close() {
    overlay.classList.remove("show");
    document.removeEventListener("keydown", onKeydown);
    clearTimeout(hideTimer);
    // Matches .modal-overlay/.modal-dialog transition duration (200ms).
    hideTimer = setTimeout(() => {
      overlay.hidden = true;
      if (logoutModalReturnFocus) {
        logoutModalReturnFocus.focus();
        logoutModalReturnFocus = null;
      }
    }, 200);
  }

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  cancelBtn.addEventListener("click", close);
  confirmBtn.addEventListener("click", () => {
    clearAuthSession();
    window.location.href = "login.html";
  });

  overlay._open = () => {
    clearTimeout(hideTimer);
    overlay.hidden = false;
    requestAnimationFrame(() => overlay.classList.add("show"));
    document.addEventListener("keydown", onKeydown);
    cancelBtn.focus();
  };

  return overlay;
}

function openLogoutModal(triggerEl) {
  logoutModalReturnFocus = triggerEl || document.activeElement;
  ensureLogoutModal()._open();
}

// Plays the cinematic "double doors" transition: the login screen covers
// itself, splits in two, and slides outward to reveal `nextPage` (already
// loading behind it in a hidden iframe) before handing off to a real
// navigation. Falls back to an instant redirect if the door markup isn't on
// the page, or the user prefers reduced motion.
function playDoorReveal(nextPage) {
  const doorLeft = document.getElementById("door-left");
  const doorRight = document.getElementById("door-right");
  const frame = document.getElementById("reveal-frame");

  if (!doorLeft || !doorRight || !frame || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
    window.location.href = nextPage;
    return;
  }

  // localStorage.displayName must already be set before this loads, or
  // auth-gated pages inside the iframe will bounce themselves to login.html.
  frame.src = nextPage;
  doorLeft.classList.add("visible");
  doorRight.classList.add("visible");

  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    window.location.href = nextPage;
  };

  // Wait for the doors to fade to fully opaque before swapping the reveal
  // frame in and sliding them apart - otherwise the destination page flashes
  // through the doors while they're still translucent mid-fade.
  setTimeout(() => {
    document.body.classList.add("door-active");
    requestAnimationFrame(() => {
      doorLeft.classList.add("open");
      doorRight.classList.add("open");
    });
    doorLeft.addEventListener("transitionend", (e) => {
      if (e.propertyName === "transform") finish();
    }, { once: true });
    setTimeout(finish, 1300);
  }, 340);
}

function loginIllustration() {
  return `
  <svg viewBox="0 0 220 220" width="100%" height="100%">
    <ellipse cx="110" cy="192" rx="70" ry="10" fill="#00000022"/>
    <rect x="52" y="26" width="80" height="150" rx="16" fill="#ffffff" opacity="0.12"/>
    <rect x="60" y="36" width="64" height="130" rx="8" fill="#ffffff" opacity="0.9"/>
    <circle cx="92" cy="80" r="16" fill="none" stroke="#2dd4bf" stroke-width="4"/>
    <rect x="82" y="92" width="20" height="16" rx="3" fill="#2dd4bf"/>
    <rect x="70" y="118" width="44" height="8" rx="4" fill="#99f6e4"/>
    <rect x="70" y="132" width="30" height="8" rx="4" fill="#99f6e4"/>
    <rect x="150" y="46" width="42" height="42" rx="10" fill="#4c6ef5" opacity="0.9"/>
    <path d="M171 58a9 9 0 0 0-9 9v2a9 9 0 0 0 18 0v-2a9 9 0 0 0-9-9Zm0 4c1.2 0 2.2.6 2.8 1.6-.4.5-1.6 1.4-2.8 1.4s-2.4-.9-2.8-1.4A3.4 3.4 0 0 1 171 62Zm0 14c-2.6 0-4.9-1.3-6.3-3.3.6-1.2 3.4-2.4 6.3-2.4s5.7 1.2 6.3 2.4A7.6 7.6 0 0 1 171 76Z" fill="#fff"/>
    <rect x="146" y="98" width="40" height="40" rx="10" fill="#7c93d1" opacity="0.9"/>
    <path d="M166 108l9 5v8c0 6-4 9-9 11-5-2-9-5-9-11v-8l9-5Z" fill="#fff"/>
    <path d="M162 118l3 3 6-6" stroke="#2dd4bf" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="42" cy="150" r="10" fill="#bcd4ee"/>
    <rect x="33" y="160" width="18" height="30" rx="7" fill="#3b5fa4"/>
    <rect x="27" y="188" width="10" height="22" rx="4" fill="#0b1b2f"/>
    <rect x="45" y="188" width="10" height="22" rx="4" fill="#0b1b2f"/>
    <rect x="24" y="168" width="10" height="20" rx="4" fill="#3b5fa4" transform="rotate(-18 24 168)"/>
    <rect x="50" y="164" width="10" height="20" rx="4" fill="#3b5fa4" transform="rotate(30 50 164)"/>
  </svg>`;
}

// Renders the login form into `container` (expected to be the .login-card element).
// `closeHref` is where the corner "x" navigates back to.
function renderLoginCard(container, closeHref = "index.html") {
  container.innerHTML = `
    <a href="${closeHref}" class="login-close" aria-label="Close">&times;</a>
    <div class="login-illustration">${loginIllustration()}</div>
    <div class="login-form-panel">
      <p class="login-welcome">Welcome Back! 👋</p>
      <h3>Login to Your Account</h3>
      <p class="login-note"></p>

      <div class="login-field">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21a8 8 0 0 0-16 0"/><circle cx="12" cy="7" r="4"/></svg>
        <input type="search" id="login-name-input" placeholder="Username" maxlength="40" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" />
      </div>
      <div class="login-field">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>
        <input type="text" id="login-password-input" class="login-password-mask" placeholder="Password" maxlength="60" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" />
      </div>

      <button class="btn btn-primary login-submit" id="login-submit-btn">Login</button>

      <p class="login-signup">Don't have an account? <a href="#" id="signup-link">Sign up for free</a></p>
    </div>
  `;

  const submit = () => {
    const name = container.querySelector("#login-name-input").value.trim();
    const password = container.querySelector("#login-password-input").value.trim();
    if (!name || !password) {
      showToast("Enter a username and password to continue.", true);
      return;
    }
    localStorage.setItem("displayName", name);
    showToast(`Welcome, ${name}!`);
    const submitBtn = container.querySelector("#login-submit-btn");
    if (submitBtn) submitBtn.disabled = true;
    // Small beat so the welcome toast registers before the door reveal starts.
    setTimeout(() => playDoorReveal(closeHref), 450);
  };

  container.querySelector("#login-submit-btn").addEventListener("click", submit);
  ["login-name-input", "login-password-input"].forEach((id) => {
    container.querySelector(`#${id}`).addEventListener("keydown", (e) => {
      if (e.key === "Enter") submit();
    });
  });

  container.querySelector("#signup-link").addEventListener("click", (e) => {
    e.preventDefault();
    showToast("No signup needed - just log in with any name and password.");
  });
}

// Renders an "already logged in" state into `container`.
function renderLoggedInCard(container, name, continueHref = "index.html") {
  container.innerHTML = `
    <a href="${continueHref}" class="login-close" aria-label="Close">&times;</a>
    <div class="login-illustration">${loginIllustration()}</div>
    <div class="login-form-panel" style="text-align:center;">
      <p class="login-welcome">You're logged in as</p>
      <h3>${escapeHtml(name)}</h3>
      <p class="login-note">This is a local demo profile stored only in this browser - no account system behind it.</p>
      <a href="${continueHref}" class="btn btn-primary login-submit" id="continue-btn" style="text-decoration:none;">Continue</a>
      <button class="btn btn-outline login-submit" id="logout-btn" style="margin-top:10px;">Logout</button>
    </div>
  `;
  container.querySelector("#continue-btn").addEventListener("click", (e) => {
    e.preventDefault();
    playDoorReveal(continueHref);
  });
  container.querySelector("#logout-btn").addEventListener("click", () => {
    clearAuthSession();
    showToast("Logged out.");
    renderLoginCard(container, continueHref);
  });
}

function initHeroParallax() {
  const heroArt = document.querySelector(".hero-art");
  const bg = document.getElementById("layer-bg");
  const fg = document.getElementById("layer-fg");
  if (!heroArt || !bg || !fg) return;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  heroArt.addEventListener("mousemove", (e) => {
    const rect = heroArt.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    bg.style.transform = `translate(${px * 10}px, ${py * 6}px)`;
    fg.style.transform = `translate(${px * 22}px, ${py * 14}px)`;
  });

  heroArt.addEventListener("mouseleave", () => {
    bg.style.transform = "";
    fg.style.transform = "";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initLoginButton();
  initThemeToggle();
  initHeroParallax();
});
