"""
change mirro from false to true
"""

import argparse
import json
from pathlib import Path

RAW_DIR = Path("raw")


DIRECTIONAL = {
    # words
    "واحد",
    "عشرة",
    "اثنان",
    "ثلاثة",
    "أربعة",
    "خمسة",
    "ستة",
    "سبعة",
    "ثمانية",
    "تسعة",
    # letters
    "ر",
    "ز",
    "و",
    "ي",
    "ل",
    "ك",
}


def fix(apply: bool = False):
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        print(f"No JSON files found in {RAW_DIR}/")
        return

    changed = 0
    skipped = 0

    for path in files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        label = d.get("label", "")
        current = d.get("mirrorable", True)

        if label in DIRECTIONAL and current:
            action = "FIXING" if apply else "WOULD FIX"
            print(f"  {action:8s}  {path.name}  ({label})  mirrorable: true → false")
            if apply:
                d["mirrorable"] = False
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=2)
            changed += 1
        else:
            skipped += 1

    print()
    if apply:
        print(f"Done. {changed} files updated, {skipped} already correct.")
        print("Now run:  python build_dataset.py  then  python train.py")
    else:
        print(f"Dry run: {changed} files would be updated, {skipped} already correct.")
        print("Run with --apply to make changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = parser.parse_args()
    fix(apply=args.apply)
