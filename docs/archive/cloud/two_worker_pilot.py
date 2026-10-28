
import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

from scenario_generator import generate_pair


def worker(task):
    try:
        result = generate_pair(**task)
        return {
            "status": "PASS",
            "result": result,
        }
    except Exception:
        return {
            "status": "FAIL",
            "error": traceback.format_exc(),
        }


def run_two_worker_pilot(tasks):
    if len(tasks) != 2:
        raise ValueError("Exactly two pilot tasks are required")

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, tasks))

    for index, result in enumerate(results, 1):
        print(f"Worker {index}: {result['status']}")
        print(result.get("result", result.get("error")))

    if any(result["status"] != "PASS" for result in results):
        raise RuntimeError("Two-worker pilot failed")

    return results
