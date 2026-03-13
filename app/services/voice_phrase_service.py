import random
from typing import List

# Array of human-like "didn't catch that" response phrases
DIDNT_CATCH_RESPONSES: List[str] = [
    "Hmm, I missed that—mind saying it again?",
    "Didn't quite get that, can you repeat?",
    "I didn't hear you clearly, would you mind repeating?",
    "Can you say that again real quick?",
    "I might've misheard—could you repeat that?",
]

# Array of follow-up phrases for when the agent didn't catch something
FOLLOW_UP_RESPONSES: List[str] = [
    "Could you repeat that for me?",
    "Mind saying that one more time?",
    "Can you try that again?",
    "Would you mind repeating that?",
    "Could you say that again?",
]


def get_random_didnt_catch_response() -> str:
    """Get a random 'didn't catch that' response to make interactions feel more human."""
    return random.choice(DIDNT_CATCH_RESPONSES)


def get_random_follow_up_response() -> str:
    """Get a random follow-up response to make interactions feel more human."""
    return random.choice(FOLLOW_UP_RESPONSES)

