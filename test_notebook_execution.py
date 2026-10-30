"""
test_notebook_execution.py
Executes all code cells in assignment3_machine_translation.ipynb in headless mode.
"""

import json
import sys
import os

# Set headless non-interactive backend for automated testing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  # mock show to prevent GUI blocking

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(PROJECT_DIR, "assignment3_machine_translation.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"[Notebook Test] Found {len(nb['cells'])} total cells in notebook.", flush=True)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"[Notebook Test] Executing {len(code_cells)} code cells sequentially...\n", flush=True)

globs = {
    "__file__": nb_path,
    "__name__": "__main__",
}

for idx, cell in enumerate(code_cells, start=1):
    code = "".join(cell["source"])
    print(f"--> Executing Cell {idx}/{len(code_cells)}...", flush=True)
    try:
        exec(code, globs)
        print(f"    [OK] Cell {idx} executed successfully.", flush=True)
    except Exception as e:
        print(f"    [ERROR] Cell {idx} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 60)
print(" NOTEBOOK VERIFICATION PASSED: All cells ran with zero errors!")
print("=" * 60, flush=True)
