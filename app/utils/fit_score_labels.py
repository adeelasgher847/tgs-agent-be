"""Human-readable labels for match scores: only Relevant vs Irrelevant + short plain summary."""

# At or above this overall score (0–1), the candidate is labeled Relevant; below, Irrelevant.
RELEVANCE_THRESHOLD = 0.5


def explain_fit_score(score: float) -> tuple[int, str, str]:
    """
    Returns:
        match_percent: 0–100 (rounded)
        fit_label: exactly "Relevant" or "Irrelevant"
        fit_summary: one plain sentence
    """
    s = max(0.0, min(1.0, float(score)))
    pct = int(round(s * 100))

    if s >= RELEVANCE_THRESHOLD:
        label = "Relevant"
        summary = (
            "Enough overlap with the job on paper that this candidate is worth considering "
            "for the next step (still use your own judgement and interviews)."
        )
    else:
        label = "Irrelevant"
        summary = (
            "Not enough overlap with this job for the system to treat it as a fit; "
            "the profile does not line up clearly with what this role needs."
        )

    return pct, label, summary
