from abc import ABCMeta, abstractmethod
from typing import Callable

import gym

from aivle_grader.abc.agent import Agent
from aivle_grader.abc.evaluator import Evaluator, EvaluationResult
from aivle_grader.abc.util import time_limiter


class TestCase(metaclass=ABCMeta):
    def __init__(
        self,
        case_id,
        time_limit: float,
        n_runs: int,
        agent_init: dict,
        env: gym.Env,
        evaluator: Evaluator,
    ):
        self.case_id = case_id
        self.time_limit = time_limit
        self.n_runs = n_runs
        self._agent_init = agent_init
        self.env = env
        self.evaluator = evaluator

    def evaluate(self, create_agent: Callable[..., Agent]) -> EvaluationResult:
        try:
            with time_limiter(self.time_limit):
                agent = create_agent(self.case_id)
                return self.run(agent)
        except Exception as e:
            return self.terminate(e)

    @abstractmethod
    def run(self, agent) -> EvaluationResult:
        pass

    def terminate(self, e) -> EvaluationResult:
        self.evaluator.terminate(e)
        return self.evaluator.get_result()
