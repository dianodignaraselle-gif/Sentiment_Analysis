// The backend (Flask) always serves the API on port 5000. When the frontend
// is opened through Flask itself (http://localhost:5000) relative "/api"
// works. When it's opened some other way instead - a VS Code Live
// Server / preview tab, or the file straight off disk - "/api" would
// resolve against that origin instead, so fall back to an absolute URL
// pointing at the Flask server (CORS is enabled on the backend for this).
const BACKEND_PORT = 5000;
const API_BASE = (() => {
  const { protocol, hostname, port } = window.location;
  if (protocol === "http:" && (port === "" || port === String(BACKEND_PORT))) {
    return "/api";
  }
  return `http://${hostname || "localhost"}:${BACKEND_PORT}/api`;
})();

async function apiRequest(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new Error(
      `Can't reach the backend at ${API_BASE}. Make sure it's running: cd backend && python app.py`
    );
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.error) message = body.error;
    } catch (_) {}
    throw new Error(message);
  }
  return res.json();
}

// Each logged-in display name only ever sees the comments it submitted -
// this keeps a brand new account's Dataset/Visualization empty instead of
// showing everyone else's (or the seeded demo) data.
//
// Normalized (trimmed + lowercased) so logging back in as "Kai" still finds
// data saved under "kai" or " Kai " - the display name shown in the UI keeps
// its original casing, only the key used to look up records is normalized.
function currentOwner() {
  return (localStorage.getItem("displayName") || "").trim().toLowerCase();
}

const Api = {
  analyze(comments, product) {
    return apiRequest("/analyze", {
      method: "POST",
      body: JSON.stringify({ comments, product, username: currentOwner() }),
    });
  },
  dataset({ page = 1, pageSize = 10, search = "", product = "" } = {}) {
    const params = new URLSearchParams({
      page,
      page_size: pageSize,
      search,
      product,
      owner: currentOwner(),
    });
    return apiRequest(`/dataset?${params.toString()}`);
  },
  products() {
    return apiRequest(`/dataset/products?owner=${encodeURIComponent(currentOwner())}`);
  },
  stats() {
    return apiRequest(`/stats?owner=${encodeURIComponent(currentOwner())}`);
  },
  exportCsvUrl() {
    return `${API_BASE}/dataset/export?owner=${encodeURIComponent(currentOwner())}`;
  },
};
