"""Minimal client for the Aether inference API.

python examples/client_example.py --url http://localhost:8000
python examples/client_example.py --stream --steps 64
"""

from __future__ import annotations

import argparse
import json

import httpx


def generate(url: str, **params: object) -> dict:  # type: ignore[type-arg]
    """One-shot generation."""
    response = httpx.post(f"{url}/generate", json=params, timeout=120)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def stream(url: str, **params: object) -> None:
    """Watch the text denoise live over server-sent events."""
    with httpx.stream("POST", f"{url}/generate/stream", json=params, timeout=120) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "step":
                    # \r keeps it on one line so the text visibly resolves.
                    print(
                        f"\r[{data['step']}/{data['total_steps']}] "
                        f"masked={data['n_masked']:4d} {data['text'][:70]!r}",
                        end="",
                        flush=True,
                    )
                elif event == "done":
                    print(f"\ndone in {data['latency_ms']:.0f} ms ({data['model_version']})")
                elif event == "error":
                    print(f"\nerror: {data['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--sampler", default="confidence", choices=["ancestral", "confidence"])
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--stream", action="store_true", help="stream the denoising steps")
    args = ap.parse_args()

    ready = httpx.get(f"{args.url}/ready", timeout=10)
    if ready.status_code != 200:
        raise SystemExit(f"server not ready: {ready.json()}")
    info = httpx.get(f"{args.url}/model", timeout=10).json()
    print(f"model {info['model_version']} ({info['params']:,} params, step {info['step']})\n")

    params = {
        "length": args.length,
        "steps": args.steps,
        "sampler": args.sampler,
        "temperature": args.temperature,
    }
    if args.stream:
        stream(args.url, **params)
        return

    result = generate(args.url, n_samples=args.n_samples, **params)
    print(
        f"nfe={result['nfe']}  latency={result['latency_ms']:.0f} ms  "
        f"batch_size={result['batch_size']}\n"
    )
    for i, text in enumerate(result["texts"], 1):
        print(f"--- sample {i} ---\n{text}\n")


if __name__ == "__main__":
    main()
