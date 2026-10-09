"""Rebuild markdown cells of the executed notebook from notebook_src without re-running anything.

Every code cell's source must be byte-identical to the executed notebook's, otherwise this refuses:
outputs are only ever carried over onto the exact code that produced them.
"""
import os
import sys

import nbformat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_notebook import OUT, build  # noqa: E402

executed = nbformat.read(OUT, as_version=4)
tmp = OUT + ".fresh"
build(tmp)
fresh = nbformat.read(tmp, as_version=4)
os.remove(tmp)
assert len(fresh.cells) == len(executed.cells), "cell count changed; re-execute instead"
for i, (f, e) in enumerate(zip(fresh.cells, executed.cells)):
    assert f.cell_type == e.cell_type, f"cell {i} type changed"
    if f.cell_type == "code":
        assert f.source == e.source, f"code cell {i} changed; re-execute instead"
        f.outputs, f.execution_count = e.outputs, e.execution_count
        f.metadata = e.metadata
fresh.metadata = executed.metadata
nbformat.validate(fresh)
nbformat.write(fresh, OUT)
print(f"refreshed markdown in {OUT}; {sum(c.cell_type == 'code' for c in fresh.cells)} code cells kept with their outputs")
