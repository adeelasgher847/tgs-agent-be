"""
Pricing Service
Calculates dynamic per-minute pricing for:
 - OpenAI / GPT models (token-based)
 - Gemini models (token-based)
 - ElevenLabs (TTS) plans (chars-based)
Also includes Twilio voice cost per minute.
"""

from typing import Optional, Dict, Any

# Constants
TOKENS_FRACTION = 200 / 1_000_000  # 0.0002 -> 200 tokens per minute as fraction of 1M
TWILIO_COST_PER_MIN = 0.0140
DEFAULT_CHARS_PER_MIN = 900  # approx chars per spoken minute

class PricingService:
    def __init__(self):
        # -----------------------------
        # OpenAI pricing (input $/1M, output $/1M) — from your table
        # -----------------------------
        self.openai = {
            "gpt-5": (1.25, 10.00),
            "gpt-5-mini": (0.25, 2.00),
            "gpt-5-nano": (0.05, 0.40),
            "gpt-5-pro": (15.00, 120.00),
            "gpt-4.1": (3.00, 12.00),
            "gpt-4.1-mini": (0.80, 3.20),
            "gpt-4.1-nano": (0.20, 0.80),
            "gpt-4o": (1.25, 5.00),
            "gpt-4o-2024-05-13": (2.50, 7.50),
            "gpt-4o-mini": (0.08, 0.30),
            "o1": (7.50, 30.00),
            "o1-pro": (75.00, 300.00),
            "o3-pro": (10.00, 40.00),
            "o3": (1.00, 4.00),
            "o3-deep-research": (5.00, 20.00),
            "o4-mini": (0.55, 2.20),
            "o4-mini-deep-research": (1.00, 4.00),
            "o3-mini": (0.55, 2.20),
            "o1-mini": (0.55, 2.20),
            "computer-use-preview": (1.50, 6.00),
        }

        # -----------------------------
        # Gemini pricing (input $/1M, output $/1M) — from your table
        # -----------------------------
        self.gemini = {
            "gemini-2.5-pro": (0.63, 5.00),
            # Google AI Studio paid tier (text/image/video): ai.google.dev/gemini-api/docs/pricing
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-2.5-flash-preview": (0.30, 2.50),
            "gemini-2.5-flash-lite": (0.05, 0.20),
            "gemini-2.0-flash": (0.05, 0.20),
            "gemini-2.0-flash-lite": (0.04, 0.15),
        }

        # -----------------------------
        # ElevenLabs plans: cost per 1K chars (USD)
        # from the table you provided
        # -----------------------------
        self.eleven_plans = {
            "free": None,        # free -> no cost per char (treated as 0)
            "starter": 0.17,
            "creator": 0.22,
            "pro": 0.20,
            "scale": 0.17,
            "business": 0.12,
        }

    # -----------------------------
    # Helpers
    # -----------------------------
    def _lookup_token_prices(self, model_name: str) -> Optional[tuple]:
        key = model_name.lower().strip()
        if key in self.openai:
            return self.openai[key]
        if key in self.gemini:
            return self.gemini[key]
        return None

    def llm_cost_per_min(self, model_name: str) -> Optional[float]:
        """
        Compute LLM cost per minute using formula:
          (input $/1M + output $/1M) * TOKENS_FRACTION
        Returns rounded float or None if model not found
        """
        prices = self._lookup_token_prices(model_name)
        if not prices:
            return None
        input_price, output_price = prices
        cost = (input_price + output_price) * TOKENS_FRACTION
        return round(cost, 6)

    def tts_cost_per_min(self, plan_name: Optional[str], chars_per_min: Optional[int] = None, turbo: bool = False) -> float:
        """
        Compute ElevenLabs TTS cost per minute from plan cost per 1K chars.
        - plan_name: e.g. 'Business' or 'creator' (case-insensitive)
        - turbo: if True, chars per minute is halved (turbo uses ~50% chars)
        Returns 0.0 for Free or unknown plan.
        """
        if not plan_name:
            return 0.0
        plan_key = plan_name.lower().strip()
        if plan_key not in self.eleven_plans:
            return 0.0
        plan_price = self.eleven_plans[plan_key]
        if plan_price is None:
            return 0.0
        chars = chars_per_min or DEFAULT_CHARS_PER_MIN
        if turbo:
            chars = chars // 2
        # plan_price is $ per 1K chars -> (plan_price / 1000) * chars
        cost = (plan_price / 1000.0) * chars
        return round(cost, 6)

    def get_pricing_for_model(self,
                              model_name: str,
                              *,
                              include_twilio: bool = True,
                              eleven_plan: Optional[str] = None,
                              tts_turbo: bool = False,
                              chars_per_min: Optional[int] = None) -> Dict[str, Optional[float]]:
        """
        Return pricing breakdown for a model.
        - model_name: the model stored in DB (e.g. 'gpt-5', 'gemini-2.5-pro', 'eleven_multilingual_v2')
        - include_twilio: whether to include twilio cost in the total (default True)
        - eleven_plan: if you want to include TTS cost, pass the ElevenLabs plan name (e.g. 'Business'). If None, TTS cost = 0.
        - tts_turbo: use turbo calculation (half chars)
        - chars_per_min: override default chars per minute for TTS
        Returns dict:
          {
            "llm_cost_per_minute": float | None,
            "tts_cost_per_minute": float,
            "twilio_cost_per_minute": float,
            "total_cost_per_minute": float | None
          }
        """
        llm = self.llm_cost_per_min(model_name)  # None if not an LLM priceable model
        tts = self.tts_cost_per_min(eleven_plan, chars_per_min, tts_turbo) if eleven_plan else 0.0
        twilio = TWILIO_COST_PER_MIN if include_twilio else 0.0

        # Compose total:
        if llm is None and tts == 0.0:
            total = None
        else:
            # if llm is None but tts exists -> total = tts + twilio
            llm_val = llm or 0.0
            total = round(llm_val + tts + twilio, 6)

        return {
            "llm_cost_per_minute": llm,
            "tts_cost_per_minute": tts,
            "twilio_cost_per_minute": twilio,
            "total_cost_per_minute": total
        }

    def get_all_known_models_pricing(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of all known LLM models (openai+gemini) and their computed llm_cost_per_minute
        plus combined total (LLM + Twilio). Useful for building a /pricing endpoint.
        """
        out = {}
        for nm, (inp, outp) in {**self.openai, **self.gemini}.items():
            llm_cost = round((inp + outp) * TOKENS_FRACTION, 6)
            total = round(llm_cost + TWILIO_COST_PER_MIN, 6)
            out[nm] = {
                "input_per_1m": inp,
                "output_per_1m": outp,
                "llm_cost_per_minute": llm_cost,
                "twilio_cost_per_minute": TWILIO_COST_PER_MIN,
                "total_cost_per_minute": total
            }
        # Add ElevenLabs plans as TTS-only entries (shows cost-per-min for sample chars)
        for plan, cost_per_1k in self.eleven_plans.items():
            if cost_per_1k is None:
                tts = 0.0
            else:
                tts = round((cost_per_1k / 1000.0) * DEFAULT_CHARS_PER_MIN, 6)
            out[f"eleven_plan:{plan}"] = {
                "input_per_1m": None,
                "output_per_1m": None,
                "llm_cost_per_minute": None,
                "tts_cost_per_minute": tts,
                "twilio_cost_per_minute": TWILIO_COST_PER_MIN,
                "total_cost_per_minute": round(tts + TWILIO_COST_PER_MIN, 6)
            }
        return out


# Singleton
pricing_service = PricingService()
