import json
import subprocess
from typing import Dict, Any

class LLMAnalyzer:
    """
    A unified Gen-AI wrapper to analyze numerical test results
    from the American PDE option pricer.
    """

    def __init__(self, model: str = "llama3", provider: str = "ollama", openai_client=None):
        self.model = model
        self.provider = provider.lower()
        self.openai_client = openai_client

    def _build_prompt(self, results: Dict[str, Any]) -> str:
        return f"""
You are a quant analyst LLM. Analyze the option pricing test output below and provide:

1. Reasonability assessment.
2. Any anomalies detected.
3. Interpretation of sensitivities (Delta, Gamma, Vega, Theta).
4. Final verdict: PASS / WARNING / FAIL.

Test Results (JSON):
{json.dumps(results, indent=4)}
"""

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

    def _run_openai(self, prompt: str) -> str:
        if self.openai_client is None:
            return "[ERROR] No OpenAI client provided."

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"[ERROR] OpenAI API failed: {str(e)}"

    def analyze_results(self, results: Dict[str, Any]) -> str:
        prompt = self._build_prompt(results)
        print("Running LLM analysis...")
    
        if self.provider == "ollama":
            print("Using Ollama provider.")
            return self._run_ollama(prompt)

        elif self.provider == "openai":
            return self._run_openai(prompt)

        else:
            return f"[ERROR] Unknown provider: {self.provider}"
