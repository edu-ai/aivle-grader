from abc import ABCMeta, abstractmethod


class EvaluationResult(object):
    def __init__(self, value, error, name: str = "score", results=None):
        if results is None:
            results = []
        self.name = name
        self.results = results
        self.value = value
        self.error = error

    def get_json(self) -> dict:
        return {
            "name": self.name,
            "results": self.results,
            "value": self.value,
            "error": self.error,
        }


class Evaluator(metaclass=ABCMeta):
    def __init__(self):
        self._error = None

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def step(self, full_state) -> None:
        pass

    @abstractmethod
    def done(self) -> None:
        pass

    def terminate(self, e: Exception) -> None:
        self._error = e
        self.done()

    @abstractmethod
    def get_result(self) -> EvaluationResult:
        pass
