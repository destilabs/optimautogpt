
import os
import dotenv
import requests
import toolz as tz

from finetoolformer.api import call_openai, call_huggingface
from finetoolformer.tasks.task_type import TaskType
from finetoolformer.tasks import get_task_by_description
from finetoolformer.tools import get_assistant_messages

dotenv.load_dotenv()


def finetune(payload="", parameters=None, options={"use_cache": False}, api_token="") -> str:
    """Calling GPT-J-6B API to ask it to generate text from a prompt.

    Args:
        payload (str, optional): Task specific prompt. Defaults to "".
        parameters (_type_, optional): Set of API specific parameters. Defaults to None.
        options (dict, optional): Request options. Defaults to {"use_cache": False}.
        api_token (str, optional): API token for huggingface inference endpoints. Defaults to "".

    Returns:
        str: Generated text.
    """
    try:
        text = call_huggingface(payload, parameters, options, api_token)
    except requests.exceptions.HTTPError as e:
        return str(e)
    else:
        return text.split("Output:")[-1].split("###")[0]

def main(inquiry):
    OPEN_AI_API_TOKEN = os.getenv("OPEN_AI_API_TOKEN")

    task_options = [t.value for t in TaskType] + ["None", "Translate"]
    task_options_tokens_len = [len(t.split(" ")) for t in task_options]
    longest_option_length = max(task_options_tokens_len)
    prompt = f"""
        Classify the problem below as one of the following product categories: ({' '.join(task_options)}):\n\n{inquiry}
    """
    parameters = {
        "model": "gpt-3.5-turbo", 
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 1, "max_tokens": int(longest_option_length * 1.4)
    }

    text = call_openai(parameters, OPEN_AI_API_TOKEN)
    text = get_assistant_messages(text.choices)[0].message.content

    task = get_task_by_description(text.replace("\n", ""))()
    
    return task.run(inquiry)

if __name__ == "__main__":
    main(inquiry="")
