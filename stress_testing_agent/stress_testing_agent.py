import json
import logging
from typing import List, Dict, Any, Optional

from .stress_scenario import stress_prompt, generate_scenarios_with_ollama
from ..Option_Pricer.PDE import AmericanPDEPricer

logger = logging.getLogger(__name__)


def generate_scenarios(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    stress_level: str = "high",
    model: str = "llama3",
    use_ollama: bool = True,
) -> List[Dict[str, float]]:
    """Generate stress scenarios using the LLM prompt helper in `stress_scenario`.

    If Ollama is not available or JSON parsing fails, falls back to a deterministic local generator.
    """
    prompt = stress_prompt(S0=S0, r=r, q=q, sigma=sigma, stress_level=stress_level)

    if use_ollama:
        try:
            scenarios = generate_scenarios_with_ollama(prompt, model=model)
            # basic validation
            if not isinstance(scenarios, list) or not all(isinstance(s, dict) for s in scenarios):
                raise ValueError("Invalid scenarios format returned from model")
            return scenarios
        except Exception as e:
            logger.warning("Ollama generation failed (%s) — falling back to local generator", e)

    # Local fallback generator (simple perturbations)
    return _local_fallback_scenarios(S0=S0, r=r, q=q, sigma=sigma, stress_level=stress_level)


def _local_fallback_scenarios(S0: float, r: float, q: float, sigma: float, stress_level: str) -> List[Dict[str, float]]:
    """Create deterministic stress scenarios when model generation is not possible.

    The function returns 12 scenarios with asymmetric downside spot shocks and larger volatility shocks.
    """
    import math

    out = []
    if stress_level.lower() in ("low", "l"):
        s0_factors = [0.98, 0.99, 1.0, 1.01, 1.02]
        sigma_factors = [0.9, 1.0, 1.1]
    elif stress_level.lower() in ("medium", "m"):
        s0_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        sigma_factors = [0.75, 1.0, 1.25]
    else:  # high
        s0_factors = [0.7, 0.8, 0.9, 1.0, 1.1]
        sigma_factors = [0.5, 0.75, 1.0, 1.5]

    for sf in s0_factors:
        for vf in sigma_factors:
            s = {
                "S0": round(float(S0 * sf), 6),
                "r": round(float(r), 6),
                "q": round(float(q), 6),
                "sigma": round(float(max(1e-6, sigma * vf)), 6),
            }
            out.append(s)
            if len(out) >= 20:
                break
        if len(out) >= 20:
            break

    # Ensure at least 10 scenarios
    if len(out) < 10:
        out = out * ((10 // len(out)) + 1)
        out = out[:12]

    return out


def price_scenarios(
    scenarios: List[Dict[str, float]],
    K: float,
    T: float,
    option_type: str = "call",
    N_S: int = 400,
    N_t: int = 400,
    theta: float = 0.5,
    return_greeks: bool = False,
) -> List[Dict[str, Any]]:
    """Price each scenario using the PDE pricer and return results with metadata.

    Returns a list where each element is the scenario dict merged with pricing output.
    """
    results: List[Dict[str, Any]] = []

    for scen in scenarios:
        S0 = float(scen.get("S0"))
        r = float(scen.get("r"))
        q = float(scen.get("q"))
        sigma = float(scen.get("sigma"))

        pricer = AmericanPDEPricer(
            S0=S0,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
            option_type=option_type,
            N_S=N_S,
            N_t=N_t,
            theta=theta,
        )

        try:
            res = pricer.price(return_grid=False)
            price = float(res.price)
            entry = {**scen, "price": price}
            if return_greeks:
                entry["greeks"] = pricer.greeks()
        except Exception as e:
            logger.exception("Pricing failed for scenario %s: %s", scen, e)
            entry = {**scen, "price": None, "error": str(e)}

        results.append(entry)

    return results


def run_and_save(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    K: float,
    T: float,
    stress_level: str = "high",
    out_file: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Generate scenarios and price them. Optionally save results to JSON file."""
    scenarios = generate_scenarios(S0=S0, r=r, q=q, sigma=sigma, stress_level=stress_level, **kwargs)
    priced = price_scenarios(scenarios, K=K, T=T)

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(priced, f, indent=2)

    return priced


if __name__ == "__main__":
    # quick CLI example
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--stress_level", type=str, default="high")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-ollama", dest="use_ollama", action="store_false")
    args = parser.parse_args()

    res = run_and_save(
        S0=args.S0,
        r=args.r,
        q=args.q,
        sigma=args.sigma,
        K=args.K,
        T=args.T,
        stress_level=args.stress_level,
        out_file=args.out,
        use_ollama=args.use_ollama,
    )

    print(json.dumps(res, indent=2))
