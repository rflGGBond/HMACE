from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, observation: Any) -> Any:
        """
        Takes an observation and returns an action.
        """
        pass
