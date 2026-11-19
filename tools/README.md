# Blackroot Tactical Tools Directory

Place pre-compiled tactical binaries here for automatic detection by the privilege escalation ladder:

- `GodPotato.exe`: Will be automatically resolved by `WindowsPrivesc._resolve_godpotato_binary()` (Tier 3 resolver).
- Or run: `python build_embed_godpotato.py tools/GodPotato.exe` to compile directly into Python base64 constant.
