from dataclasses import dataclass

@dataclass(frozen=True)
class Step:
    text_id: int
    project: int
    situation: int
    topic: int
    correct_action: int
    revealed_action: int
