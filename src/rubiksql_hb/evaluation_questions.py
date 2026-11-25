"""Closed, content-pinned evaluation inputs, including the archived evolution set."""
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionSet:
    """Repository-relative input path, original byte digest and complete row count."""

    path: str
    digest: str
    count: int


QUESTION_SETS = {
    "330": QuestionSet("resources/evolution/questions_330.json",
                       "sha256:319922936681d61917a0ec8b456536a51e1bc005ccf9ca2e2a1d2866ac42a5da", 330),
    "minidev500": QuestionSet("resources/official_input/minidev_500_20240627.json",
                              "sha256:def4b2b43a9b06955193418f24c9be170eb6d83763ab702311790e0cdda8c791", 500),
    "dev1534": QuestionSet("resources/official_input/dev_20240627.json",
                           "sha256:0a8a29722d4fbe13415f467365cf40811c5524b3327f3b3ccc11a85e1beaeb8c", 1534),
}

EVOLUTION_SET = QUESTION_SETS["minidev500"]
