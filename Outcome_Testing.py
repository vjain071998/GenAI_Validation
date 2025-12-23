'''import sys
import os
import json

from Option_Pricer.PDE import AmericanPDEPricer
from outcome_analysis_agent.OutcomeAnalysisAgent import OutcomeAgent
from outcome_analysis_agent.generate_spot_scenarios import spot_scenerio

from LLMs.analysis_llm import LLMAnalyzer
import json

base_params = {
    "S0": 100,
    "K": 100,
    "r": 0.05,
    "q": 0.00,
    "sigma": 0.20,
    "T": 1.0,
    "N_S": 200,
    "N_t": 200,
    "option_type":"call"
}

# STEP 1: Create LLM + Agent
llm = LLMAnalyzer()
print(llm)
#pricer = AmericanPDEPricer(**base_params)
agent = OutcomeAgent(llm, AmericanPDEPricer, base_params)

# STEP 2: Generate Spot-only scenarios
scenarios = spot_scenerio(base_spot=100, pct_range=0.20, step=0.05)
print(scenarios)
# STEP 3: Run analysis
results = agent.run_property_test(
    scenarios=scenarios,
    property_name="S0",
)

output_dir = os.getcwd()  

# 1. Save raw numerical results
results_path = os.path.join(output_dir, "OutcomeResults.txt")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


summary = agent.analyze_results(results, "S0", base_params["option_type"])
output_dir = os.getcwd()   # script directory


# 2. Save LLM summary
summary_path = os.path.join(output_dir, "OutputSummary.txt")
with open(summary_path, "w") as f:
    f.write(summary)
print("Summary saved to:", summary_path)
'''

import os
import json

from Option_Pricer.PDE import AmericanPDEPricer
from outcome_analysis_agent.OutcomeAnalysisAgent import OutcomeAgent

from outcome_analysis_agent.generate_spot_scenarios import (
    spot_scenerio,
    strike_scenerio,
    rate_scenerio,
    vol_scenerio,
    maturity_scenerio,
)

from LLMs.analysis_llm import LLMAnalyzer

# =========================
# Base Parameters
# =========================
base_params = {
    "S0": 100,
    "K": 100,
    "r": 0.05,
    "q": 0.00,
    "sigma": 0.20,
    "T": 1.0,
    "N_S": 200,
    "N_t": 200,
    "option_type": "call",
}

# =========================
# Create LLM + Agent
# =========================
llm = LLMAnalyzer()
agent = OutcomeAgent(llm, AmericanPDEPricer, base_params)

output_dir = os.getcwd()
option_type = base_params["option_type"]

# ============================================================
# Scenario registry (single source of truth)
# ============================================================
TESTS = {
    "S0": spot_scenerio(base_params["S0"]),
    "K": strike_scenerio(base_params["K"]),
    "r": rate_scenerio(base_params["r"]),
    "sigma": vol_scenerio(base_params["sigma"]),
    "T": maturity_scenerio(base_params["T"]),
}

# ============================================================
# Run all outcome-analysis tests
# ============================================================
for property_name, scenarios in TESTS.items():

    print(f"Running outcome analysis for {property_name}")

    results = agent.run_property_test(
        scenarios=scenarios,
        property_name=property_name,
    )

    # Save raw numerical results
    with open(
        os.path.join(output_dir, f"OutcomeResults_{property_name}.txt"), "w"
    ) as f:
        json.dump(results, f, indent=4)

    # LLM analysis
    summary = agent.analyze_results(
        results,
        property_name,
        option_type,
    )

    with open(
        os.path.join(output_dir, f"OutputSummary_{property_name}.txt"), "w"
    ) as f:
        f.write(summary)

print("All outcome analysis tests completed successfully.")
