import logging
from multiprocessing import Process, Queue

import gym
import ma_gym
from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import EnvSerializer
from aivle_gym.judge_multi_env import JudgeMultiEnv

from aivle_grader.abc.agent import Agent
from aivle_grader.evaluator import RewardEvaluator
from aivle_grader.test_case import ReinforcementLearningTestCase
from aivle_grader.test_suite import TestSuite


class PongJudgeEnv(JudgeMultiEnv):
    def __init__(self):
        self.env = gym.make("PongDuel-v0")
        super().__init__(
            PongEnvSerializer(),
            self.env.action_space,
            self.env.observation_space,
            self.env.reward_range,
            self.env.n_agents,
            {0: 0, 1: 1},
        )

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        pass  # TODO

    def seed(self, seed=None):
        pass  # TODO


class PongEnvSerializer(EnvSerializer):
    def action_to_json(self, action):
        return action

    def json_to_action(self, action_json):
        return action_json

    def observation_to_json(self, obs):
        return obs

    def json_to_observation(self, obs_json):
        return obs_json

    def info_to_json(self, info):
        return info

    def json_to_info(self, info_json):
        return info_json


class PongAgentEnv(AgentEnv):
    def __init__(self, uid):
        base_env = gym.make("PongDuel-v0")
        super().__init__(
            PongEnvSerializer(),
            base_env.action_space[0],
            base_env.observation_space[0],
            base_env.reward_range,
            uid=uid,
        )


class PongAgent(Agent):
    def __init__(self):
        self.action_space = gym.make("PongDuel-v0").action_space[0]

    def step(self, state):
        return self.action_space.sample()

    def reset(self):
        pass


def create_agent(case_id, *args, **kwargs):
    return PongAgent()


def execute(uid, q: Queue):
    n_runs = 10
    evaluator = RewardEvaluator()
    agent_env = PongAgentEnv(uid=uid)
    seeds = []
    tc = ReinforcementLearningTestCase(
        t_max=10000,
        env=agent_env,
        evaluator=evaluator,
        agent_init={},
        seeds=seeds,
        case_id=0,
        time_limit=3600,
        n_runs=n_runs,
    )
    ts = TestSuite(suite_id=f"pong_{uid}", cases=[tc])
    result = ts.run(create_agent)
    q.put({"uid": uid, "result": result})


def main():
    logging.basicConfig(level=logging.INFO)

    judge_env = PongJudgeEnv()
    judge_proc = Process(target=judge_env.start, args=())
    judge_proc.start()
    q = Queue()
    try:
        ts_0_proc = Process(target=execute, args=(0, q))
        ts_1_proc = Process(target=execute, args=(1, q))
        ts_0_proc.start()
        ts_1_proc.start()
        ts_0_proc.join()
        ts_1_proc.join()
        while not q.empty():
            print(q.get())
    except Exception as e:
        print(e)
    finally:
        judge_proc.terminate()
        judge_env.close()


if __name__ == "__main__":
    main()
