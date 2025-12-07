
WEIGHTS = {
    "sleeping": -10,
    "yawning": -5,
    "using_phone": -20,
    "turning_around": -15,
    "raise_hand": +10,
    "note_taking": +5
}

def calculate_simple_score(logs):
    """
    logs: list of documents, each doc has 'behavior' key
    returns: (score, category)
    """
    total = 0
    for l in logs:
        b = l.get("behavior")
        total += WEIGHTS.get(b, 0)

    if total < 0:
        category = "poor"
    elif total < 10:
        category = "average"
    else:
        category = "good"

    return total, category
