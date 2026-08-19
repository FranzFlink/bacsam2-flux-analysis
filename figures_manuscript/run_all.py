"""Run every manuscript figure script in this folder (fig*.py, figA*.py)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
failed = []
for script in sorted(HERE.glob("fig*.py")):
    print(f"== {script.name}")
    if subprocess.run([sys.executable, str(script)]).returncode != 0:
        failed.append(script.name)
print("FAILED: " + ", ".join(failed) if failed else "all figures OK")
sys.exit(1 if failed else 0)
