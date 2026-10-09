"""Assemble training_loop.ipynb from notebook_src/notebook_source.py.

Cells are separated by '#%% md' / '#%% code' lines. '@@NAME@@' markers are replaced by the contents of
notebook_src/parts/NAME.(md|py) when such a file exists.
"""
import os
import re
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "notebook_source.py")
OUT = os.path.join(os.path.dirname(HERE), "training_loop.ipynb")


def fill(text):
    def sub(m):
        name = m.group(1)
        for ext in (".md", ".py"):
            p = os.path.join(HERE, "parts", name + ext)
            if os.path.exists(p):
                return open(p, encoding="utf-8").read().strip("\n")
        return m.group(0)
    return re.sub(r"@@([A-Z_]+)@@", sub, text)


def build(out=OUT):
    src = open(SRC, encoding="utf-8").read()
    parts = re.split(r"^#%% (md|code)\s*$", src, flags=re.M)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, body in zip(parts[1::2], parts[2::2]):
        body = fill(body.strip("\n"))
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    left = [c for c in nb.cells if c.cell_type == "code" and "@@" in c.source]
    assert not left, "unfilled code placeholder"
    nbformat.validate(nb)
    nbformat.write(nb, out)
    print(f"wrote {out}: {len(nb.cells)} cells")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else OUT)
