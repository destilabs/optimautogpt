
from dataclasses import dataclass
from typing import Dict

@dataclass
class OpenAiResponse:
    """OpenAI response object.
    """
    id: str
    object: str
    created: int
    model: str
    choices: list
    usage: Dict[str, int]

    def __post_init__(self):
        self.choices = [OpenAiChoice(**choice) for choice in self.choices]

@dataclass
class OpenAiMessage:
    """OpenAI message object.
    """
    role: str
    content: str
    
@dataclass
class OpenAiChoice:
    """OpenAI choice object.
    """
    message: OpenAiMessage
    finish_reason: str
    index: int

    def __post_init__(self):
        self.message = OpenAiMessage(**self.message)
