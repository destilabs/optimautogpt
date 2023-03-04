import os
from typing import Any
from finetoolformer.tasks.abstract_task import AbstractTask
from finetoolformer.api import call_openai
from finetoolformer.tasks.task_type import TaskType
from finetoolformer.tools import get_assistant_messages

class CalendarTask(AbstractTask):
    def __init__(self, verbosity=1) -> None:
        super().__init__()
        self.task_description = TaskType.Calendar.value
        self.verbosity = verbosity
        self.input_mask = "[INPUT]"
        self.task_prompt = f"""
            Your task is to add calls to a Calendar API to a piece of text. 
            The API calls should help you get information required to complete the text. 
            You can call the API by writing "[Calendar()]" 
            Here are some examples of API calls:
            Input: Today is the first Friday of the year.
            Output: Today is the first [Calendar()] Friday of the year.
            ###
            Input: The president of the United States is Joe Biden.
            Output: The president of the United States is [Calendar()] Joe Biden.
            ###
            Input: The current day of the week is Wednesday.
            Output: The current day of the week is [Calendar()] Wednesday.
            ###
            Input: The number of days from now until Christmas is 30.
            Output: The number of days from now until Christmas is [Calendar()] 30.
            ###
            Input: The store is never open on the weekend, so today it is closed.
            Output: The store is never open on the weekend, so today [Calendar()] it is closed.
            ###
            Input: {self.input_mask}
            Output: [Calendar("
        """

    def run(self, inquiry:str) -> Any:
        prompt = self.compile_task_prompt(inquiry)

        if self.verbosity > 0:
            print(prompt)

        parameters = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 1
        }

        text = call_openai(parameters, os.getenv("OPEN_AI_API_TOKEN"))
        text = get_assistant_messages(text.choices)[0].message.content

        return text


    def compile_task_prompt(self, prompt: str) -> str:
        prompt = self.task_prompt.replace(self.input_mask, prompt)

        parameters = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 1
        }

        task_prompt = call_openai(parameters, os.getenv("OPEN_AI_API_TOKEN"))

        return task_prompt.choices[0].message.content