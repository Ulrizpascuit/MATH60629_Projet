"""E0 — Référence recalculée sur la machine de l'étude.

Relance tout le pipeline de l'ancien projet (sans run_data.py : les données de
Data/ sont figées) dans une copie temporaire du dépôt, puis range les résultats
dans experiments/E0_reference_mac/. Les dossiers outputs/ et Data/Models/ du
dépôt, qui contiennent la référence publiée, ne sont jamais modifiés.

Usage : .venv/bin/python Code/experiments/e0_reference_mac.py   (≈ 8 min sur 10 cœurs)
"""
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "experiments" / "E0_reference_mac"
# Ce qu'il faut copier pour faire tourner le pipeline (chemins calculés depuis Function/)
TO_COPY = ["Code", "Function", "Data", "run_strategy.py", "run_comparison.py"]
STEPS = [
    ["run_strategy.py"],
    ["run_comparison.py"],
    ["Code/mixt_features.py"],
    ["Code/ml_analysis.py"],
]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "numba", "joblib", "matplotlib"]


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="e0_reference_mac_") as tmp:
        work = Path(tmp)
        for name in TO_COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, work / name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, work / name)
        (work / "latex" / "tables").mkdir(parents=True)
        (work / "latex" / "figures").mkdir(parents=True)

        logs = {}
        for step in STEPS:
            t = time.time()
            res = subprocess.run([sys.executable, *step], cwd=work, capture_output=True, text=True)
            logs[step[0]] = {"seconds": round(time.time() - t, 1), "returncode": res.returncode}
            if res.returncode != 0:
                print(res.stdout[-3000:], res.stderr[-3000:])
                raise SystemExit(f"Échec de {step[0]}")

        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(work / "outputs", DEST / "outputs")
        shutil.copytree(work / "latex", DEST / "latex")
        (DEST / "models").mkdir()
        shutil.copy2(work / "Data" / "Models" / "models.pkl", DEST / "models" / "models.pkl")

    env = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "steps": logs,
        "total_seconds": round(time.time() - t0, 1),
    }
    (DEST / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    print(f"Référence recalculée dans {DEST} ({env['total_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
