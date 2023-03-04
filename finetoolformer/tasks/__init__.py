from typing import Type
from .qna_task import QnATask
from .calculator_task import CalculatorTask
from .calendar_task import CalendarTask
from .abstract_task import AbstractTask

def get_task_by_description(description: str) -> Type[AbstractTask]:
    config = {
        "Question and Answering": QnATask,
        "Calculator": CalculatorTask,
        "Calendar": CalendarTask
    }

    try:
        task = next(iter([v for k, v in config.items() if k.startswith(description)]))
    except StopIteration:
        raise ValueError(f"Task {description} not found.")
    
    return task