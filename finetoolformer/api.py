from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Dict

import requests

from finetoolformer.constants import HF_API_URL_TEMPLATE, OPEN_AI_URL

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


def call_huggingface(payload="", parameters=None, options={"use_cache": False}, api_token="") -> str:
    API_URL = str.format(HF_API_URL_TEMPLATE, model_id="EleutherAI/gpt-j-6B")
    headers = {"Authorization": f"Bearer {api_token}"}
    body = {"inputs": payload, "parameters": parameters, "options": options}

    response = requests.request("POST", API_URL, headers=headers, data=json.dumps(body))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError("Error:" + " ".join(response.json()["error"]))
    else:
        return response.json()
    
def call_openai(parameters, api_token="") -> OpenAiResponse:
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    response = requests.request("POST", OPEN_AI_URL, headers=headers, data=json.dumps(parameters))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError("Error:" + " ".join(response.json()["error"]))
    else:
        return OpenAiResponse(**response.json())
