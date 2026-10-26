# -*- coding: utf-8 -*-
"""Validate all code cells in the notebook for syntax errors."""
import json

with open("MaintenX_Predictive_Maintenance.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
total = len(cells)
code_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"Total cells : {total}")
print(f"Code cells  : {len(code_cells)}")

errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    try:
        compile(src, f"<cell {i}>", "exec")
    except SyntaxError as e:
        errors.append(f"Cell {i}: {e}")

if errors:
    print("\nSYNTAX ERRORS:")
    for e in errors:
        print(" ", e)
else:
    print("\nAll code cells pass syntax check. Notebook is ready.")
