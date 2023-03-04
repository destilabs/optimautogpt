import os
from typing import Any
from finetoolformer.tasks.abstract_task import AbstractTask
from finetoolformer.api import call_openai
from finetoolformer.tasks.task_type import TaskType
from finetoolformer.tools import get_assistant_messages

class CalculatorTask(AbstractTask):
    def __init__(self, verbosity=1) -> None:
        super().__init__()
        self.task_description = TaskType.Calculator.value
        self.verbosity = verbosity
        self.input_mask = "[INPUT]"
        self.task_prompt = f"""
            Your task is to add calls to a Calculator API to a piece of text.
            The calls should help you get information required to complete the text. 
            You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. 
            Here are some examples of API calls:
            ###
            Input: The number in the next term is 18 + 12 x 3 = 54.
            Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.
            ###
            Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
            Output: The population is 658,893 people. This is 11.4% of the national average of [Calculator(658,893 / 11.4%)] 5,763,868 people.
            ###
            Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
            Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
            ###
            Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
            Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)] 17 years.
            ###
            Input: From this, we have 4 * 30 minutes = 120 minutes.
            Output: From this, we have 4 * 30 minutes = [Calculator(4 * 30)] 120 minutes.
            ###
            Input: {self.input_mask}
            Output: [Calculator("
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