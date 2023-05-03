import os
from typing import Any
from finetoolformer.tasks.abstract_task import AbstractTask
from finetoolformer.api import call_openai
from finetoolformer.tasks.task_type import TaskType
from finetoolformer.tools import get_assistant_messages

class GroceryTask(AbstractTask):
    def __init__(self, verbosity=1) -> None:
        super().__init__()
        self.task_description = TaskType.Grocery.value
        self.verbosity = verbosity
        self.input_mask = "[INPUT]"
        self.task_prompt = f"""
            Твоїм завданням є сформувати список продуктів на основі страви, що згадано в тексті.
            Щоб додати продукти в корзину, використовуйте команду "[Grocery()]". 
            Ось деякі приклади викликів API:
            Вхідний текст: Сьогодні вечеря - куряча котлета з картоплею.
            Вихідний текст: Сьогодні вечеря - куряча [Grocery(філе куряче)] котлета з [Grocery(картопля молода)].
            ###
            Вхідний текст: Приготую на сніданок сирники.
            Вихідний текст: Рецепт сирників: [Grocery(молоко)] [Grocery(яйця)] [Grocery(сіль)] [Grocery(цукор)] [Grocery(мука)].
            ###
            Вхідний текст: Приготую на вечерю пасту.
            Вихідний текст: Рецепт пасти: [Grocery(макарони)] [Grocery(масло)] [Grocery(сіль)] [Grocery(перець чорний)] [Grocery(паста)].
            ###
            Вхідний текст: Перед залом потрібно з'їсти грецький салат.
            Вихідний текст: [Grocery(огірок)] [Grocery(помідор)] [Grocery(масло оливкове)] [Grocery(сіль)] [Grocery(перець чорний)].
            ###
            Input: {self.input_mask}
            Output: 
        """

    def run(self, inquiry:str) -> Any:
        prompt = self.compile_task_prompt(inquiry)

        if self.verbosity > 0:
            print(prompt)

        parameters = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0
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