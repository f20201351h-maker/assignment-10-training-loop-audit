"""
Execute training_loop.ipynb top-to-bottom on one Modal GPU and bring back the executed notebook,
artifacts/ and figures/.

    modal run modal_run.py                       # full run, writes into this folder
    modal run modal_run.py --out-dir pilot_out   # write somewhere else (e.g. a pilot run)

App name and output paths are specific to this project; no Modal Volume is used, so nothing is shared with
other apps in the workspace. The notebook is executed by nbclient in a fresh kernel inside a fresh container.
"""
import io
import os
import tarfile
import time

import modal

APP_NAME = "training-loop-audit"
GPU = "A10G"  # Modal's A10 option; nvidia-smi reports "NVIDIA A10G"
HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/root/trainloop"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")  # triton needs a C compiler for torch.compile
    .pip_install("torch==2.5.1", "numpy==2.1.3", "pandas==2.2.3", "matplotlib==3.9.2",
                 "nbformat==5.10.4", "nbclient==0.10.0", "ipykernel==6.29.5")
    .add_local_dir(os.path.join(HERE, "s10lab"), f"{REMOTE}/s10lab", ignore=["__pycache__"])
)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=3 * 3600, cpu=4, memory=16384)
def execute(nb_json: str, mode: str = "full") -> dict:
    import subprocess
    import nbformat
    from nbclient import NotebookClient

    os.chdir(REMOTE)
    os.environ["S10_MODE"] = mode
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    nb = nbformat.reads(nb_json, as_version=4)
    t0 = time.time()
    err = None
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": REMOTE}}).execute()
    except Exception as e:  # keep the partially executed notebook for debugging
        err = repr(e)[-4000:]
    wall = time.time() - t0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("artifacts", "figures"):
            if os.path.isdir(d):
                tar.add(d)
    return {"notebook": nbformat.writes(nb), "outputs_tgz": buf.getvalue(), "error": err,
            "execute_seconds": wall, "nvidia_smi": smi}


@app.local_entrypoint()
def main(out_dir: str = ".", mode: str = "full"):
    import json
    import sys
    sys.path.insert(0, os.path.join(HERE, "notebook_src"))
    from build_notebook import build

    nb_path = build()
    t0 = time.time()
    res = execute.remote(open(nb_path, encoding="utf-8").read(), mode)
    call_seconds = time.time() - t0
    out = os.path.join(HERE, out_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "training_loop.ipynb"), "w", encoding="utf-8") as f:
        f.write(res["notebook"])
    with tarfile.open(fileobj=io.BytesIO(res["outputs_tgz"]), mode="r:gz") as tar:
        tar.extractall(out)
    run = {"app": APP_NAME, "gpu_requested": GPU, "mode": mode, "notebook_execute_seconds": res["execute_seconds"],
           "remote_call_seconds_incl_container_start": call_seconds,
           # modal billing rates (2026-10-02): A10G $1.10/h, CPU $0.0473/core/h, memory $0.008/GiB/h
           "approx_cost_usd": round(call_seconds / 3600 * (1.10 + 4 * 0.0473 + 16 * 0.008), 3),
           "error": res["error"]}
    os.makedirs(os.path.join(out, "artifacts"), exist_ok=True)
    with open(os.path.join(out, "artifacts", "modal_run.json"), "w") as f:
        json.dump(run, f, indent=2)
    with open(os.path.join(out, "artifacts", "nvidia_smi.txt"), "w") as f:
        f.write(res["nvidia_smi"])
    print(json.dumps(run, indent=2))
    if res["error"]:
        raise SystemExit("notebook execution failed: " + res["error"][-1500:])
