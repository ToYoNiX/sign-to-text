"""
Assembles the static GitHub Pages site into _site/.

  python build.py

Inputs (repo root):
  templates/index.html  — unified browser UI template
  templates/style.css   — stylesheet
  model.onnx            — exported SVM pipeline
  label_map.json        — index → Arabic label

Output:
  _site/
    index.html       (site mode: CONFIG injected, ONNX inference)
    style.css
    model.onnx
    label_map.json

GitHub Actions runs this on every push that touches these files.
"""

import shutil
from pathlib import Path

ROOT      = Path(__file__).parent
SITE      = ROOT / "_site"
TEMPLATES = ROOT / "templates"

TEMPLATE  = TEMPLATES / "index.html"
CSS       = TEMPLATES / "style.css"
MODEL     = ROOT / "model.onnx"
LABELMAP  = ROOT / "label_map.json"

def build():
    for src in (TEMPLATE, CSS, MODEL, LABELMAP):
        if not src.exists():
            raise FileNotFoundError(
                f"Missing: {src}\n"
                "Run `python export_onnx.py` first to generate model.onnx and label_map.json."
            )

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()

    html = TEMPLATE.read_text(encoding="utf-8")
    (SITE / "index.html").write_text(
        html.replace("%%CONFIG%%", '{"mode":"site"}'),
        encoding="utf-8",
    )
    shutil.copy(CSS,      SITE / "style.css")
    shutil.copy(MODEL,    SITE / "model.onnx")
    shutil.copy(LABELMAP, SITE / "label_map.json")

    print(f"Built → {SITE}/")
    for f in sorted(SITE.iterdir()):
        kb = f.stat().st_size / 1024
        print(f"  {f.name:<20} {kb:>7.1f} KB")

if __name__ == "__main__":
    build()
