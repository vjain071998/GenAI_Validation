import sys
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
    "N_t": 200
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
    option_type="Call"
)

output_dir = os.getcwd()  

# 1. Save raw numerical results
results_path = os.path.join(output_dir, "OutcomeResults.txt")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

'''
summary = agent.analyze_results(results, "S0", "Call")
output_dir = os.getcwd()   # script directory


# 2. Save LLM summary
summary_path = os.path.join(output_dir, "OutputSummary.txt")
with open(summary_path, "w") as f:
    f.write(summary)
'''