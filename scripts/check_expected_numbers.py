#!/usr/bin/env python3
"""Compare the numbers printed by the number scripts with expected_numbers.txt.

expected_numbers.txt is the committed output of manuscript_numbers.py (text and
Table 4), ec_quality.py (Appendix A), robustness_checks.py and
ec_band_sensitivity.py (robustness checks asked for in review) for this version
of the repository. Numbers are compared with a tolerance of 0.02 (one
unit in the last printed digit plus rounding) so that harmless floating-point
differences between operating systems do not raise a false alarm.

Run:  pixi run check
Update the reference after an intended change:
      pixi run python scripts/check_expected_numbers.py --update
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPECTED = REPO / "expected_numbers.txt"
NUM = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")
TOL = 0.02

out = "".join(subprocess.run([sys.executable, str(REPO / "scripts" / script)],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
              for script in ("manuscript_numbers.py", "ec_quality.py", "robustness_checks.py", "ec_band_sensitivity.py"))

if "--update" in sys.argv:
    EXPECTED.write_text(out, encoding="utf-8", newline="\n")
    print(f"updated {EXPECTED.name}")
    sys.exit(0)

exp_lines = EXPECTED.read_text(encoding="utf-8").splitlines()
new_lines = out.splitlines()
problems = []
if len(exp_lines) != len(new_lines):
    problems.append(f"line count differs: expected {len(exp_lines)}, got {len(new_lines)}")
for i, (e, n) in enumerate(zip(exp_lines, new_lines), 1):
    if NUM.sub("#", e) != NUM.sub("#", n):
        problems.append(f"line {i}: text differs\n   expected: {e}\n   got:      {n}")
        continue
    for a, b in zip(NUM.findall(e), NUM.findall(n)):
        if abs(float(a) - float(b)) > TOL:
            problems.append(f"line {i}: {a} != {b}\n   expected: {e}\n   got:      {n}")
            break

print(out)
print("=" * 72)
if problems:
    print("MISMATCH with expected_numbers.txt:")
    print("\n".join(problems))
    sys.exit(1)
print("RESULT: all numbers match expected_numbers.txt")
