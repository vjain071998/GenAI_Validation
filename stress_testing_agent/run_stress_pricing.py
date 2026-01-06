"""Run stress scenarios (via Ollama) and price each scenario with the PDE pricer.

Usage:
    python run_stress_pricing.py --S0 100 --r 0.05 --q 0.0 --sigma 0.2 --stress_level "high_vol" --K 100 --T 1.0

This script will:
 - Build a prompt using `stress_scenario.stress_prompt`
 - Call `generate_scenarios_with_ollama` to get JSON scenarios
 - For each scenario, instantiate `AmericanPDEPricer` and compute price and greeks
 - Save results to `stress_results.json` (by default)
"""

from __future__ import annotations

import json
import argparse
import datetime
from typing import List, Dict, Any

try:
    # When used as a package (recommended)
    from . import stress_scenario
    from ..Option_Pricer.PDE import AmericanPDEPricer
except Exception:
    # Fallback when executed directly as a script: add package root to sys.path
    import os
    import sys

    PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)

    import stress_scenario
    from Option_Pricer.PDE import AmericanPDEPricer


def price_stress_scenarios(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    stress_level: str,
    K: float,
    T: float,
    option_type: str = "call",
    model: str = "llama3",
    pricer_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Generate stress scenarios and price them.

    Returns a list of dicts with the scenario and pricing output.
    """
    if pricer_kwargs is None:
        pricer_kwargs = {"N_S": 400, "N_t": 400, "theta": 0.5}

    prompt = stress_scenario.stress_prompt(S0, r, q, sigma, stress_level)

    # call Ollama model (may raise subprocess errors or JSON decode errors)
    raw_scenarios = stress_scenario.generate_scenarios_with_ollama(prompt, model=model)

    if not isinstance(raw_scenarios, list):
        raise ValueError("Expected list of scenarios from LLM")

    results = []
    for idx, sc in enumerate(raw_scenarios):
        # validate minimal keys
        for key in ("S0", "r", "q", "sigma"):
            if key not in sc:
                raise KeyError(f"Scenario {idx} missing required key: {key}")

        # create pricer with scenario parameters but user-provided K, T, option_type
        pricer = AmericanPDEPricer(
            S0=float(sc["S0"]),
            K=float(K),
            r=float(sc["r"]),
            q=float(sc["q"]),
            sigma=float(sc["sigma"]),
            T=float(T),
            option_type=option_type,
            **pricer_kwargs,
        )

        price_res = pricer.price(return_grid=False)
        # compute greeks (may be somewhat expensive)
        try:
            greeks = pricer.greeks()
        except Exception:
            greeks = {"error": "greeks_failed"}

        results.append({
            "scenario_index": idx,
            "scenario": sc,
            "price": float(price_res.price),
            "greeks": greeks,
            "metadata": price_res.info,
        })

    return results


def save_results(results: List[Dict[str, Any]], out_path: str) -> None:
    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "count": len(results),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_parser():
    p = argparse.ArgumentParser(description="Run stress scenario pricing via Ollama + PDE pricer")
    p.add_argument("--S0", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--stress_level", type=str, default="high_vol")
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--option_type", type=str, choices=("put", "call"), default="call")
    p.add_argument("--model", type=str, default="llama3")
    p.add_argument("--out", type=str, default="stress_results.json")
    return p


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("Generating stress scenarios and pricing...")
    try:
        results = price_stress_scenarios(
            S0=args.S0,
            r=args.r,
            q=args.q,
            sigma=args.sigma,
            stress_level=args.stress_level,
            K=args.K,
            T=args.T,
            option_type=args.option_type,
            model=args.model,
        )

        save_results(results, args.out)
        print(f"Saved {len(results)} results to: {args.out}")
        return 0

    except Exception as e:
        print("Error during stress pricing:", str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
