"""
STT Endpointing & Premature Finalization Benchmark Tool.

Evaluates representative conversational speech patterns with varying intra-utterance
pause lengths across different endpointing thresholds (900ms, 600ms, 500ms, 400ms, 300ms)
to empirically determine:
  1. Premature finalization / false cut-off rate (%)
  2. Endpointing delay (ms from acoustic speech end to final emit)
  3. Turn-taking responsiveness vs interruption resilience
"""
from __future__ import annotations

import dataclasses
import json
from typing import List


@dataclasses.dataclass
class UtteranceProfile:
    id: str
    category: str
    description: str
    # List of (phrase_text, duration_sec, following_pause_sec)
    segments: List[tuple[str, float, float]]

    @property
    def total_speech_sec(self) -> float:
        return sum(s[1] for s in self.segments)

    @property
    def max_internal_pause_sec(self) -> float:
        if len(self.segments) <= 1:
            return 0.0
        return max(s[2] for s in self.segments[:-1])

    @property
    def full_text(self) -> str:
        return " ".join(s[0] for s in self.segments)


# Representative dataset of conversational voice speech profiles
BENCHMARK_PROFILES: List[UtteranceProfile] = [
    # 1. Short Fast Confirmations
    UtteranceProfile(
        id="fast_conf_1",
        category="short_confirmation",
        description="Quick 'Yes please' with immediate silence",
        segments=[("Yes please", 0.6, 0.0)],
    ),
    UtteranceProfile(
        id="fast_conf_2",
        category="short_confirmation",
        description="Quick 'That works for me' without pauses",
        segments=[("That works for me", 1.1, 0.0)],
    ),
    # 2. Natural Mid-Sentence Pauses & Hesitations
    UtteranceProfile(
        id="hesitation_1",
        category="natural_pause",
        description="Natural thinking pause: 'I'd like to book... [450ms pause] ...for tomorrow afternoon'",
        segments=[
            ("I'd like to book", 1.2, 0.45),
            ("for tomorrow afternoon", 1.4, 0.0),
        ],
    ),
    UtteranceProfile(
        id="hesitation_2",
        category="natural_pause",
        description="Longer hesitation: 'Can you check... [550ms pause] ...if Dr. Smith is available?'",
        segments=[
            ("Can you check", 0.9, 0.55),
            ("if Dr. Smith is available?", 1.5, 0.0),
        ],
    ),
    UtteranceProfile(
        id="hesitation_3",
        category="natural_pause",
        description="Deep pause: 'Well... [650ms pause] ...I am not entirely sure about that.'",
        segments=[
            ("Well", 0.4, 0.65),
            ("I am not entirely sure about that", 1.8, 0.0),
        ],
    ),
    # 3. Long Continuous Speech / Explanations
    UtteranceProfile(
        id="long_query_1",
        category="complex_query",
        description="Continuous explanation of consultation requirements",
        segments=[
            ("I am calling to understand your standard pricing and whether consultations are covered by insurance", 4.2, 0.0)
        ],
    ),
    # 4. Numbers, Phone Digits & Alphanumerics
    UtteranceProfile(
        id="digits_1",
        category="digits_and_codes",
        description="Digit grouping with 350ms rhythmic pauses: 'My number is 415... [350ms] ...555... [350ms] ...0199'",
        segments=[
            ("My number is 415", 1.4, 0.35),
            ("555", 0.8, 0.35),
            ("0199", 0.9, 0.0),
        ],
    ),
    UtteranceProfile(
        id="digits_2",
        category="digits_and_codes",
        description="Slow digit reading with 520ms pauses: 'Reference code 9... [520ms] ...4... [520ms] ...28'",
        segments=[
            ("Reference code 9", 1.3, 0.52),
            ("4", 0.5, 0.52),
            ("28", 0.7, 0.0),
        ],
    ),
    # 5. Question with Self-Correction
    UtteranceProfile(
        id="correction_1",
        category="self_correction",
        description="Self-correction with pause: 'Let us do Wednesday... [480ms] ...actually Thursday is better'",
        segments=[
            ("Let us do Wednesday", 1.3, 0.48),
            ("actually Thursday is better", 1.6, 0.0),
        ],
    ),
]


def evaluate_endpointing_threshold(
    profiles: List[UtteranceProfile],
    endpointing_ms: int,
) -> dict:
    """
    Simulate STT acoustic energy endpointing on benchmark utterance profiles.
    
    A premature finalization occurs whenever an internal pause between words in the same
    intended utterance exceeds the endpointing threshold, causing the STT engine to prematurely
    commit an incomplete fragment and trigger an agent response mid-sentence.
    """
    threshold_sec = endpointing_ms / 1000.0
    total_utterances = len(profiles)
    premature_cuts = 0
    endpointing_delays_ms = []
    cut_details = []

    for profile in profiles:
        # Check if any intra-sentence pause exceeded the endpointing threshold
        was_cut_prematurely = False
        for seg_idx, (phrase, dur, pause) in enumerate(profile.segments[:-1]):
            if pause >= threshold_sec:
                was_cut_prematurely = True
                cut_details.append({
                    "profile_id": profile.id,
                    "category": profile.category,
                    "pause_sec": pause,
                    "threshold_sec": threshold_sec,
                    "premature_fragment": phrase,
                    "missed_fragment": profile.segments[seg_idx + 1][0],
                })
                break

        if was_cut_prematurely:
            premature_cuts += 1

        # The endpointing delay is the silence required after the final word
        endpointing_delays_ms.append(endpointing_ms)

    cut_rate_pct = (premature_cuts / total_utterances) * 100.0
    return {
        "endpointing_ms": endpointing_ms,
        "total_test_utterances": total_utterances,
        "premature_cuts_count": premature_cuts,
        "premature_cut_rate_pct": round(cut_rate_pct, 1),
        "endpointing_delay_ms": endpointing_ms,
        "cut_details": cut_details,
    }


def run_stt_endpointing_benchmark() -> dict:
    thresholds = [900, 600, 500, 400, 300]
    results = {}
    for th in thresholds:
        results[f"{th}ms"] = evaluate_endpointing_threshold(BENCHMARK_PROFILES, th)
    return results


if __name__ == "__main__":
    report = run_stt_endpointing_benchmark()
    print(json.dumps(report, indent=2))
