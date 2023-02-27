from collections import defaultdict
import json

import requests

from finetoolformer.constants import HF_API_URL_TEMPLATE, OPEN_AI_URL


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
    
def call_openai(parameters, api_token="") -> str:
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    response = requests.request("POST", OPEN_AI_URL, headers=headers, data=json.dumps(parameters))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError("Error:" + " ".join(response.json()["error"]))
    else:
        return response.json()
