import sys
import os
import json

# ---------------------------------------------
# Fix Python path so imports work
# ---------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Option_Pricer.PDE import AmericanPDEPricer
from LLMs.analysis_llm import LLMAnalyzer


# ---------------------------------------------
# Utility: Greeks via bump-and-revalue
# ---------------------------------------------
def compute_greeks(pricer, bump=0.01):
    base = pricer.price(return_grid=False)
    base_price = base.price

    # ----- Delta: bump S0 -----
    bump_size = pricer.S0 * bump
    pr_up = pricer._price_with_override(S0=pricer.S0 + bump_size)
    pr_down = pricer._price_with_override(S0=pricer.S0 - bump_size)
    delta = (pr_up - pr_down) / (2 * bump_size)

    # ----- Gamma -----
    gamma = (pr_up - 2 * base_price + pr_down) / (bump_size ** 2)

    # ----- Vega: bump sigma -----
    sig_bump = pricer.sigma * bump
    pr_up_sig = pricer._price_with_override(sigma=pricer.sigma + sig_bump)
    pr_down_sig = pricer._price_with_override(sigma=pricer.sigma - sig_bump)
    vega = (pr_up_sig - pr_down_sig) / (2 * sig_bump)

    # ----- Theta: bump T -----
    T_bump = max(1e-6, pricer.T * bump)
    pr_theta = pricer._price_with_override(T=pricer.T - T_bump)
    theta = (pr_theta - base_price) / (-T_bump)

    return {
        "price": base_price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }



# ---------------------------------------------
# Main Testing Pipeline
# ---------------------------------------------
def main():

    print("\nRunning American PDE Option Pricer...\n")

    # Create pricer instance
    pricer = AmericanPDEPricer(
        S0=100,
        K=100,
        r=0.05,
        q=0.00,
        sigma=0.20,
        T=1.0,
        N_S=200,
        N_t =200
    )

    # Compute Greeks using bump-and-revalue
    results = compute_greeks(pricer)

    print("Raw Numerical Output:")
    print(json.dumps(results, indent=4))

    # LLM analysis
    analyzer = LLMAnalyzer(model="llama3")   # or gpt-4.1, finbert, mistral etc.

    summary = analyzer.analyze_results(results)

    print("\n\n==================== GEN-AI ANALYSIS SUMMARY ====================\n")
    print(summary)
    print("\n=================================================================\n")


# ---------------------------------------------
if __name__ == "__main__":
    main()
