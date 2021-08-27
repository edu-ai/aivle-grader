import logging
import sys
import time
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
        super().__init__(PongEnvSerializer(), self.env.action_space, self.env.observation_space, self.env.reward_range,
                         self.env.n_agents, {0: 0, 1: 1})

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
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
        super().__init__(PongEnvSerializer(), base_env.action_space[0], base_env.observation_space[0],
                         base_env.reward_range, uid=uid)


class PongAgent(Agent):
    def __init__(self):
        self.action_space = gym.make("PongDuel-v0").action_space[0]

    def step(self, state):
        return self.action_space.sample()

    def reset(self):
        pass


def create_agent(case_id, *args, **kwargs):
    return PongAgent()


def run0():
    n_runs = 10
    evaluator = RewardEvaluator()
    agent_env_0 = PongAgentEnv(uid=0)
    seeds = []
    tc_0_0 = ReinforcementLearningTestCase(
        t_max=10000,
        env=agent_env_0,
        evaluator=evaluator,
        agent_init={},
        seeds=seeds,
        case_id=0,
        time_limit=3600,
        n_runs=n_runs
    )
    ts_0 = TestSuite(suite_id="pong_0", cases=[tc_0_0])
    ts_0.run(create_agent)


def run1():
    n_runs = 10
    evaluator = RewardEvaluator()
    agent_env_0 = PongAgentEnv(uid=1)
    seeds = []
    tc_0_0 = ReinforcementLearningTestCase(
        t_max=10000,
        env=agent_env_0,
        evaluator=evaluator,
        agent_init={},
        seeds=seeds,
        case_id=0,
        time_limit=3600,
        n_runs=n_runs
    )
    ts_0 = TestSuite(suite_id="pong_1", cases=[tc_0_0])
    ts_0.run(create_agent)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # judge_env = PongJudgeEnv()
    # judge_proc = Process(target=judge_env.start, args=())
    # judge_proc.start()
    ts_0_proc = None
    ts_1_proc = None
    try:
        ts_0_proc = Process(target=run0)
        ts_1_proc = Process(target=run1)
        ts_0_proc.start()
        ts_1_proc.start()
        # ts_0_proc.join()
        # ts_1_proc.join()
    except Exception as e:
        print(e)
    finally:
        # print(judge_proc.exitcode)
        # judge_proc.terminate()
        if ts_0_proc is not None:
            print(ts_0_proc.exitcode)
            ts_0_proc.terminate()
        if ts_1_proc is not None:
            print(ts_1_proc.exitcode)
            ts_1_proc.terminate()


def main2():
    n_runs = 10
    evaluator = RewardEvaluator()
    agent_env_0 = PongAgentEnv(uid=0)
    seeds = [23333 for _ in range(n_runs)]
    tc_0_0 = ReinforcementLearningTestCase(
        t_max=10000,
        env=agent_env_0,
        evaluator=evaluator,
        agent_init={},
        seeds=seeds,
        case_id=0,
        time_limit=3600,
        n_runs=n_runs
    )
    ts_0 = TestSuite(suite_id="pong_0", cases=[tc_0_0])
    ts_0.run(create_agent)


def main3():
    n_runs = 10
    evaluator = RewardEvaluator()
    agent_env_0 = PongAgentEnv(uid=1)
    seeds = [23333 for _ in range(n_runs)]
    tc_0_0 = ReinforcementLearningTestCase(
        t_max=10000,
        env=agent_env_0,
        evaluator=evaluator,
        agent_init={},
        seeds=seeds,
        case_id=0,
        time_limit=3600,
        n_runs=n_runs
    )
    ts_0 = TestSuite(suite_id="pong_1", cases=[tc_0_0])
    ts_0.run(create_agent)


def main4():
    judge_env = PongJudgeEnv()
    judge_proc = Process(target=judge_env.start, args=())
    judge_proc.start()
    env_0 = PongAgentEnv(uid=0)
    env_1 = PongAgentEnv(uid=1)
    evaluators = [RewardEvaluator() for _ in range(2)]
    tc = MultiAgentTestCase(case_id="multi_tc_0", time_limit=10.0, n_runs=2, n_agents=2, envs=[env_0, env_1],
                            evaluators=evaluators, t_max=10000, agent_init={})
    create_agents = [create_agent for _ in range(2)]
    tc.evaluate(create_agents)


if __name__ == "__main__":
    run1()
