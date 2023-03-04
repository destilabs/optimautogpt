
import os
import dotenv

from finetoolformer.api import call_openai
from finetoolformer.tasks.task_type import TaskType
from finetoolformer.tasks import get_task_by_description
from finetoolformer.tools import get_assistant_messages

dotenv.load_dotenv()


def pipeline(inquiry):
    OPEN_AI_API_TOKEN = os.getenv("OPEN_AI_API_TOKEN")

    task_options = [t.value for t in TaskType]
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

