def estimate_severity(confidence: float) -> dict:
    """
    Estimates disease severity from model confidence score.
    Higher confidence = clearer disease presentation = Mild.
    Lower confidence = ambiguous / advanced = Severe.
    """
    if confidence >= 0.85:
        severity = "Mild"
        score = 0.25
    elif confidence >= 0.60:
        severity = "Moderate"
        score = 0.55
    else:
        severity = "Severe"
        score = 0.85

    return {
        "severity": severity,
        "severity_score": score,
        "confidence": confidence
    }