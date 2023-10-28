from typing import List

import gym

from aivle_grader.abc.evaluator import Evaluator
from aivle_grader.abc.test_case import TestCase

## DO NOT CHANGE THIS CODE
def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out

class ElevatorTestCase(TestCase):
    """Custom test case for grid driving task"""

    def __init__(
            self,
            t_max: int,
            seeds: List[int],
            case_id,
            time_limit: float,
            n_runs: int,
            env: gym.Env,
            disc2state,
            evaluator: Evaluator,
            agent_init: dict = None,
            is_render=False
    ):
        super().__init__(case_id, time_limit, n_runs, agent_init, env, evaluator)
        self.is_render = is_render
        self.t_max = t_max
        self.seeds = seeds
        self.use_seed = len(self.seeds) > 0
        self.disc2state = disc2state

        if self.use_seed:
            assert (
                    len(self.seeds) == self.n_runs
            )  # provide fixed random seed for each episode

    def run(self, agent):
        for i_episode in range(self.n_runs):
            if self.use_seed:
                # this is not the correct implementation of seeds in Gym env
                # but for backward compatible purposes I followed how Rizki did it...
                self.env.random_seed = self.seeds[i_episode]
            state = self.env.reset()
            env_features = list(self.env.observation_space.keys())
            # agent.reset()
            self.evaluator.reset()
            for t in range(self.t_max):
                state_desc = self.disc2state(int(state))
                state_list = convert_state_to_list(state_desc, env_features)
                action = agent.step(state_list)
                next_state, reward, done, info = self.env.step(action)  # self.env.step(action)
                if self.is_render:
                    self.env.render(state)
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
                # total_reward += reward
                print()
                # print(f'step       = {step}')
                print(f'state      = {state}')
                print(f'action     = {action}')
                print(f'next state = {next_state}')
                print(f'reward     = {reward}')
                state = next_state
                if done:
                    break
        return self.evaluator.get_result()
