import json
from copy import deepcopy
import subprocess

class OutcomeAgent:
    """
    GenAI Agent #1:
    Evaluates pricing results under controlled parameter bumps and
    uses an LLM (Ollama) to check if results behave as expected.
    """

    def __init__(self, llm_client, pricer_class, base_params, model: str = "llama3", provider: str = "ollama", openai_client=None):
        self.llm = llm_client
        self.pricer_class = pricer_class
        self.base_params = base_params
        self.model = model
        self.provider = provider.lower()
        self.openai_client = openai_client
        

    def _run_ollama(self, prompt: str) -> str:
        cmd = ["ollama", "run", self.model]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"[ERROR] Ollama execution failed: {e.stderr}"
        
    # ---------------------------------------------------------
    # MAIN RUNNER
    # ---------------------------------------------------------
    def run_property_test(self, scenarios, property_name):
        """
        Evaluates the behaviour of ONE property only (e.g. Spot).
        property_name: "S0", "K", "r", "sigma", etc.
        scenarios: list of dicts with bumped property values
        """
        results_collection = []
        scenarios = [{}] + scenarios

        for i, sc in enumerate(scenarios):
            params = deepcopy(self.base_params)
            params.update(sc)

            pricer = self.pricer_class(**params)
            out = pricer._price_with_override(**sc)

            is_base_case = (sc == {})

            results_collection.append({
                "scenario_id": "Base Case" if is_base_case else i,
                "changed_property": property_name,
                "scenario_params": sc,
                "full_params": params,
                "pricing_output": out
            })

        return results_collection

    def analyze_results(self, results, property_name, option_type) -> str:
        prompt = self._prepare_prompt(results, property_name, option_type)
        print("Running LLM analysis...")
    
        if self.provider == "ollama":
            print("Using Ollama provider.")
            return self._run_ollama(prompt)

        elif self.provider == "openai":
            return self._run_openai(prompt)

        else:
            return f"[ERROR] Unknown provider: {self.provider}"
        
    # ---------------------------------------------------------
    # LLM PROMPT CREATION
    # ---------------------------------------------------------
    def _prepare_prompt(self, results, property_name, option_type):
        prompt = f"""
You are a quantitative finance validator. 
We are testing the effect of changing ONE property only: '{property_name}'.
The option type is '{option_type}' (you must infer expected behaviour logically).
Your task:

For EACH scenario:
1. Compare the bumped value vs base case.
2. Check whether the pricing output behaves logically.
3. Decide PASS / FAIL / DON'T KNOW.
4. Explain in one short sentence.

Output STRICTLY in JSON list:
[
  {{ "scenario_id": 1, "verdict": "Pass|Fail|Don't know", "reason": "..." }},
  ...
]

Below is the data:
"""

        prompt += json.dumps(results, indent=2)
        return prompt

    # ---------------------------------------------------------
    # SAFE PARSER
    # ---------------------------------------------------------
    def _parse_llm_json(self, text, expected_items):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        # fallback – all “Don't know”
        return [
            {
                "scenario_id": i+1,
                "verdict": "Don't know",
                "reason": "LLM output not parseable"
            }
            for i in range(expected_items)
        ]
