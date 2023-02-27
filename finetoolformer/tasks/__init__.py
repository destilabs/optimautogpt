from typing import Type
from .qna_task import QnATask
from .abstract_task import AbstractTask

def get_task_by_description(description: str) -> Type[AbstractTask]:
    config = {
        "Question and Answering": QnATask
    }

    try:
        task = next(iter([v for k, v in config.items() if k.startswith(description)]))
    except StopIteration:
        raise ValueError(f"Task {description} not found.")
    
    return task