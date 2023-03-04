import json

import requests

from typing import Optional

from finetoolformer.constants import HF_API_URL_TEMPLATE, OPEN_AI_URL
from finetoolformer.dto import OpenAiResponse


def call_huggingface(
    payload: Optional[str] = "",
    parameters: Optional[dict] = None,
    options: Optional[dict] = {"use_cache": False},
    api_token: Optional[str] = "",
) -> str:
    """
    Calling GPT-J-6B API to ask it to generate text from a prompt.

    Args:
        payload (str, optional): Task specific prompt. Defaults to "".
        parameters (_type_, optional): Set of API specific parameters. Defaults to None.
        options (dict, optional): Request options. Defaults to {"use_cache": False}.
        api_token (str, optional): API token for huggingface inference endpoints. Defaults to "".

    Returns:
        str: Generated text.
    """
    API_URL = str.format(HF_API_URL_TEMPLATE, model_id="EleutherAI/gpt-j-6B")
    headers = {"Authorization": f"Bearer {api_token}"}
    body = {"inputs": payload, "parameters": parameters, "options": options}

    response = requests.request("POST", API_URL, headers=headers, data=json.dumps(body))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError(
            "Error:" + " ".join(response.json()["error"])
        )
    else:
        return response.json()


def call_openai(
        parameters: Optional[dict] = None, 
        api_token: Optional[str] = "") -> OpenAiResponse:
    """
    Calling OpenAI API to ask it to generate text from a prompt.

    Args:
        parameters (dict): Set of API specific parameters.
        api_token (str, optional): API token for OpenAI API. Defaults to "".

    Returns:
        OpenAiResponse: OpenAI response object.
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.request(
        "POST", OPEN_AI_URL, headers=headers, data=json.dumps(parameters)
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError(
            "Error:" + " ".join(response.json()["error"])
        )
    else:
        return OpenAiResponse(**response.json())
