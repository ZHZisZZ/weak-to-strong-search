from abc import ABC
from dataclasses import dataclass
from typing import Union, Text, List


@dataclass
class EvaluatorInput:
    prompt: Text
    response: Text


class BaseEvaluator(ABC):

    def eval(self, input: Union[EvaluatorInput, List[EvaluatorInput]]) -> List[float]:
        raise NotImplementedError
