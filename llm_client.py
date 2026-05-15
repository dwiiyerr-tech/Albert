"""
Provider-neutral LLM client.

Supported providers:
- anthropic: native Anthropic Messages API
- openai-compatible: any /chat/completions API using Bearer auth
- mock: deterministic offline responses for smoke tests only
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from config import LLM_BASE_URL, LLM_ENSEMBLE, LLM_MODEL, LLM_PROVIDER, require_llm_api_key

logger = logging.getLogger(__name__)


@dataclass
class LLMToolCall:
    name: str
    input: dict[str, Any]


class LLMClient:
    """Small adapter over Anthropic and OpenAI-compatible chat APIs."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = (provider or LLM_PROVIDER or "anthropic").lower()
        self.model = model or LLM_MODEL
        self.api_key = api_key or require_llm_api_key()
        self.base_url = (base_url if base_url is not None else LLM_BASE_URL).rstrip("/")
        self._anthropic_client = None

        if self.provider == "mock":
            return
        if self.provider == "anthropic":
            try:
                import anthropic
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "LLM_PROVIDER=anthropic requires the 'anthropic' package. "
                    "Install requirements or use LLM_PROVIDER=openai-compatible."
                ) from exc
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                **({"base_url": self.base_url} if self.base_url else {}),
            )
        elif self.provider in {"openai", "openai-compatible", "compatible"}:
            if not self.base_url:
                self.base_url = "https://api.openai.com/v1"
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER={self.provider!r}. "
                "Use 'anthropic', 'openai-compatible', or 'mock'."
            )

    @classmethod
    def ensemble(cls) -> list["LLMClient"]:
        """Return configured ensemble clients, or the primary client if none."""
        if not LLM_ENSEMBLE:
            return [cls()]
        clients = []
        for raw in LLM_ENSEMBLE.split(","):
            parts = [p.strip() for p in raw.split(":", 2)]
            if not parts or not parts[0]:
                continue
            provider = parts[0]
            model = parts[1] if len(parts) > 1 and parts[1] else None
            base_url = parts[2] if len(parts) > 2 and parts[2] else None
            clients.append(cls(provider=provider, model=model, base_url=base_url))
        return clients or [cls()]

    def text(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """Return plain assistant text."""
        if self.provider == "mock":
            return self._mock_text(system=system, messages=messages)
        if self.provider == "anthropic":
            response = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return self._anthropic_text(response)

        payload = {
            "model": self.model,
            "messages": self._openai_messages(system, messages),
            "max_tokens": max_tokens,
        }
        data = self._post_openai_chat(payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Invalid OpenAI-compatible response: {data}") from exc

    def tool_calls(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> list[LLMToolCall]:
        """
        Ask the model to return tool calls as JSON.

        This avoids provider-specific function-calling differences and works
        across Anthropic, OpenAI-compatible hosted APIs, and local gateways.
        """
        tool_spec = json.dumps(tools, indent=2)
        system = f"""You are a tool-calling reasoning model.
Return ONLY valid JSON in this exact shape:
{{
  "tool_calls": [
    {{"name": "tool_name", "input": {{"field": "value"}}}}
  ]
}}
Use only these tools and their schemas:
{tool_spec}
If no tool call is warranted, return {{"tool_calls": []}}."""
        text = self.text(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return self._parse_tool_calls(text)

    def _post_openai_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        out = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _anthropic_text(response: Any) -> str:
        chunks = []
        for block in getattr(response, "content", []) or []:
            if hasattr(block, "text"):
                chunks.append(block.text)
        return "\n".join(chunks)

    @staticmethod
    def _parse_tool_calls(text: str) -> list[LLMToolCall]:
        data: Optional[dict[str, Any]] = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = None
        if not data:
            logger.warning("Could not parse tool-call JSON from LLM response: %s", text[:500])
            return []
        raw_calls = data.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            return []
        calls = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            inputs = call.get("input", {})
            if isinstance(name, str) and isinstance(inputs, dict):
                calls.append(LLMToolCall(name=name, input=inputs))
        return calls

    @staticmethod
    def _mock_text(*, system: str, messages: list[dict[str, Any]]) -> str:
        prompt = system + "\n" + "\n".join(str(m.get("content", "")) for m in messages)
        lower = prompt.lower()

        def _forecast_consensus() -> float:
            match = re.search(r"Consensus:\s*([\d.-]+)°F", prompt)
            return float(match.group(1)) if match else 72.0

        def _forecast_spread() -> float:
            match = re.search(r"Model spread:\s*([\d.-]+)°F", prompt)
            return max(2.5, float(match.group(1))) if match else 4.0

        def _bucket_bounds() -> tuple[float, float]:
            bucket_match = re.search(r"Temperature bucket under analysis:\s*([^\n]+)", prompt)
            bucket = bucket_match.group(1).strip() if bucket_match else ""
            below = re.search(r"below\s+([\d.-]+)°?F", bucket, re.I)
            if below:
                return float("-inf"), float(below.group(1))
            above = re.search(r"above\s+([\d.-]+)°?F", bucket, re.I)
            if above:
                return float(above.group(1)), float("inf")
            rng = re.search(r"([\d.-]+)°?F\s*[–-]\s*([\d.-]+)°?F", bucket)
            if rng:
                return float(rng.group(1)), float(rng.group(2))
            return float("-inf"), float("inf")

        def _normal_cdf(x: float, mean: float, sigma: float) -> float:
            return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))

        def _prob_for_temp(mean: float) -> float:
            lo, hi = _bucket_bounds()
            sigma = max(2.5, _forecast_spread() / 2.0)
            if lo == float("-inf") and hi == float("inf"):
                return 0.98
            if lo == float("-inf"):
                p = _normal_cdf(hi, mean, sigma)
            elif hi == float("inf"):
                p = 1.0 - _normal_cdf(lo, mean, sigma)
            else:
                p = _normal_cdf(hi, mean, sigma) - _normal_cdf(lo, mean, sigma)
            return max(0.01, min(0.99, p))

        if "tool_calls" in lower:
            return '{"tool_calls": []}'
        if '"scenarios"' in prompt or "generate the scenarios" in lower:
            consensus = _forecast_consensus()
            spread = _forecast_spread()
            return json.dumps({
                "scenarios": [
                    {
                        "name": "Model consensus",
                        "probability": 0.65,
                        "expected_temp_f": round(consensus, 1),
                        "description": "Deterministic mock scenario for smoke testing.",
                    },
                    {
                        "name": "Warmer tail",
                        "probability": 0.35,
                        "expected_temp_f": round(consensus + max(3.0, spread), 1),
                        "description": "Deterministic warm-tail scenario for smoke testing.",
                    },
                ]
            })
        if "conditional_probs" in lower:
            temps = [
                float(t)
                for t in re.findall(r"Expected temp:\s*([\d.-]+)°F", prompt)
            ]
            probs = [_prob_for_temp(t) for t in temps] or [_prob_for_temp(_forecast_consensus())]
            return json.dumps({
                "conditional_probs": probs,
                "reasoning": "Mock forecast-derived conditional probabilities.",
            })
        if "suggested_adjustment" in lower or "bayesian synthesizer" in lower:
            return '{"confidence": "medium", "reasoning": "Mock synthesis for smoke testing.", "suggested_adjustment": 0.0}'
        if '{"p":' in prompt:
            return f'Mock classic debate response.\n{{"p": {_prob_for_temp(_forecast_consensus()):.3f}}}'
        return "Mock LLM response for offline smoke testing."
