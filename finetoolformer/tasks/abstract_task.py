import abc
from typing import Any

class AbstractTask(metaclass=abc.ABCMeta):
    """Abstract class for all tasks."""
    @abc.abstractmethod
    def run(self, inquiry:str) -> Any:
        """Run the task."""
        pass

    @abc.abstractmethod
    def compile_task_prompt(self, prompt: str) -> str:
        """Compile the tuning prompt."""
        pass
