from abc import ABCMeta, abstractmethod
from typing import Callable

import gym

from aivle_grader.abc.agent import Agent
from aivle_grader.abc.evaluator import Evaluator, EvaluationResult
from aivle_grader.abc.util import time_limiter


class TestCase(metaclass=ABCMeta):
    """Abstract base class for test cases.

    There are 6 properties that need initialization:

        case_id
        time_limit
        n_runs: number of episodes to run
        agent_init (TODO): init params passed to __init__ method of Agent
        env: OpenAI Gym compatible environment
        evaluator: Evaluator object
    """
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
        """Runs `env` with provided agent for `n_runs` times under `time_limit`
        with `evaluator` attached.

        :param create_agent: a function that returns an Agent
        :return: EvaluationResult
        """
        try:
            with time_limiter(self.time_limit):
                agent = create_agent(self.case_id)
                return self.run(agent)
        except Exception as e:
            return self._terminate(e)

    @abstractmethod
    def run(self, agent) -> EvaluationResult:
        """Runs `env` with `agent` for `n_runs` times with `evaluator` attached.

        :param agent: an Agent
        :return: EvaluationResult
        """
        pass

    def _terminate(self, e) -> EvaluationResult:
        self.evaluator.terminate(e)
        return self.evaluator.get_result()
