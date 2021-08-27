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
        agent_init: init params passed to __init__ method of Agent
        env: OpenAI Gym compatible environment
        evaluator: Evaluator object
    """

    def evaluate(self, create_agent: Callable[..., Agent]) -> EvaluationResult:
        """Runs `env` with provided agent for `n_runs` times under `time_limit`
        with `evaluator` attached.

        :param create_agent: a function that returns an Agent
        :return: EvaluationResult
        """
        try:
            with time_limiter(self.time_limit):
                agent = create_agent(self.case_id, **self._agent_init)
                return self.run(agent)
        except Exception as e:
            return self._terminate(e)

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
        self._agent_init = {} if agent_init is None else agent_init
        self.env = env
        self.evaluator = evaluator

    @abstractmethod
    def run(self, agent) -> EvaluationResult:
        """Runs `env` with `agent` for `n_runs` times with `evaluator` attached.

        :param agent: an Agent
        :return: EvaluationResult
        """
        pass

    def _terminate(self, e: Exception) -> EvaluationResult:
        self.evaluator.terminate(e)
        return self.evaluator.get_result()


"""
class MultiAgentTestCase:
    def __init__(self, case_id,
                 time_limit: float,
                 n_runs: int,
                 n_agents: int,
                 agent_init: dict,
                 envs: List[AgentEnv],
                 evaluators: List[Evaluator],
                 t_max: int):
        self.case_id = case_id
        self.time_limit = time_limit
        self.n_runs = n_runs
        self._agent_init = {} if agent_init is None else agent_init
        self.evaluators = evaluators
        self.envs = envs
        self.n_agents = n_agents
        self.t_max = t_max

    def evaluate(self, create_agents: List[Callable[..., Agent]]):
        # TODO: time limit
        agents = [f(self.case_id, **self._agent_init) for f in create_agents]
        return self.run(agents)

    def run(self, agents: List[Agent]) -> List[EvaluationResult]:
        assert len(agents) == self.n_agents
        for i_episode in range(self.n_runs):
            state = []
            dones = [False for _ in range(self.n_agents)]
            for idx in range(self.n_agents):
                self.evaluators[idx].reset()
                agents[idx].reset()
                state.append(self.envs[idx].reset())
            for t in range(self.t_max):
                for idx in range(self.n_agents):
                    if dones[idx]:
                        continue
                    action = agents[idx].step(state[idx])
                    next_state, reward, done, info = self.envs[idx].step(action)
                    self.evaluators[idx].step(
                        {
                            "state": state[idx],
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                            "done": done,
                            "info": info,
                            "episode_count": i_episode,
                            "t": t,
                        }
                    )
                    state[idx] = next_state
                    if done:
                        dones[idx] = True
        return [self.evaluators[i].get_result() for i in range(self.n_agents)]

    def _terminate(self, e: Exception) -> List[EvaluationResult]:
        for i in range(self.n_agents):
            self.evaluators[i].terminate(e)
        return [self.evaluators[i].get_result() for i in range(self.n_agents)]
"""
