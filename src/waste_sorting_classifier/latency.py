"""Measure end-to-end /predict latency against a running API (e.g. after make docker-run).

python -m waste_sorting_classifier.latency [URL]   default http://localhost:8000

Uploads tests/example.jpg sequentially after a warm-up and writes reports/latency.json.
"""

import json
import math
import platform
import statistics
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime

from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path

N_REQUESTS = 50
N_WARMUP = 5


def multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def timed_request(url: str, body: bytes, content_type: str) -> float:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    body, content_type = multipart(
        "file", "example.jpg", project_path("tests/example.jpg").read_bytes()
    )
    for _ in range(N_WARMUP):
        timed_request(f"{base}/predict", body, content_type)
    ms = sorted(timed_request(f"{base}/predict", body, content_type) for _ in range(N_REQUESTS))
    with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
        health = json.load(resp)
    result = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "client_machine": f"{platform.system()} {platform.machine()}",
        "image": "tests/example.jpg (512x384 JPEG)",
        "requests": N_REQUESTS,
        "mean_ms": statistics.mean(ms),
        "p50_ms": statistics.median(ms),
        "p95_ms": ms[math.ceil(0.95 * len(ms)) - 1],  # nearest-rank: 48th of 50
        "max_ms": ms[-1],
        "model": health["model"],
        "onnx_sha256": health["onnx_sha256"],
    }
    out = project_path(load_config()["paths"]["reports_dir"]) / "latency.json"
    write_json(out, result)
    print(f"mean {result['mean_ms']:.1f} ms, p50 {result['p50_ms']:.1f} ms, "
          f"p95 {result['p95_ms']:.1f} ms over {N_REQUESTS} uploads -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
