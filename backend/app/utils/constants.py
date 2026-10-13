"""
Standard constants and disfluency lexicons for fluency analysis.
"""

from typing import Dict, List, Tuple

# Single-token vocal disfluencies (pure hesitations / fillers)
VOCAL_FILLERS = {
    "um": "vocal_disfluency",
    "uh": "vocal_disfluency",
    "erm": "vocal_disfluency",
    "er": "vocal_disfluency",
    "ah": "vocal_disfluency",
}

# Single-token discourse markers that may function as fillers or legitimate grammatical words
POTENTIAL_FILLERS = {
    "like": "potential_filler",
    "so": "potential_filler",
    "actually": "potential_filler",
    "basically": "potential_filler",
    "literally": "potential_filler",
}

# Multi-word conversational filler idioms
MULTIWORD_FILLERS: List[Tuple[Tuple[str, ...], str]] = [
    (("you", "know"), "conversational_filler"),
    (("i", "mean"), "conversational_filler"),
    (("sort", "of"), "conversational_filler"),
    (("kind", "of"), "conversational_filler"),
]

# Standard pause threshold boundaries (in seconds)
MIN_PAUSE_THRESHOLD_SECONDS: float = 0.25
SIGNIFICANT_PAUSE_SECONDS: float = 0.50
EXTENDED_PAUSE_SECONDS: float = 2.00
