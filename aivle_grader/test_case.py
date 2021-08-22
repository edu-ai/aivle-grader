from abc import ABCMeta, abstractmethod
from typing import List, Callable, Any

import gym

from aivle_grader.agent import Agent
from aivle_grader.evaluator import Evaluator, EvaluationResult
from aivle_grader.util import time_limiter


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


class ReinforcementLearningTestCase(TestCase):
    def __init__(
        self,
        t_max: int,
        seeds: List[int],
        case_id,
        time_limit: float,
        n_runs: int,
        agent_init: dict,
        env: gym.Env,
        evaluator: Evaluator,
    ):
        super().__init__(case_id, time_limit, n_runs, agent_init, env, evaluator)
        self.t_max = t_max
        self.seeds = seeds
        self.use_seed = len(self.seeds) > 0
        if self.use_seed:
            assert (
                len(self.seeds) == self.n_runs
            )  # provide fixed random seed for each episode

    def run(self, agent):
        self.evaluator.reset()
        for i_episode in range(self.n_runs):
            self.evaluator.run()
            state = self.env.reset()
            if self.use_seed:
                self.env.seed(self.seeds[i_episode])
            agent.reset()
            for t in range(self.t_max):
                action = agent.step(state)
                next_state, reward, done, info = self.env.step(action)
                self.evaluator.step(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                        "info": info,
                        "episode_count": i_episode,
                        "t": t,
                    }
                )
                state = next_state
                if done:
                    break
        self.evaluator.done()
        return self.evaluator.get_result()
