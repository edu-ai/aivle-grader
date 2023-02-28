import multiprocessing
import time
from abc import ABC
from multiprocessing import Process, Queue

from gym.envs.toy_text import taxi

from aivle_grader.taxi_test_case import TaxiTestCase
from aivle_gym.judge_env import JudgeEnv
from aivle_gym.taxi_agent import TaxiAgentEnv

from aivle_grader.evaluator import StepCountEvaluator
from aivle_grader.test_suite import TestSuite
from aivle_gym.env_serializer import SampleSerializer


def run_judge(return_queue: Queue):
    judge_env = TaxiJudgeEnv()
    return_queue.put(judge_env.bind())
    judge_env.start()


class TaxiJudgeEnv(JudgeEnv, ABC):
    def __init__(self):
        # env_info = ExampleManager.GetEnvInfo(ENV)
        # set up the environment class, choose instance 0 because every example has at least one example instance
        self.env = taxi.TaxiEnv(render_mode="ansi")
        super().__init__(
            SampleSerializer(),
            self.env.action_space,
            self.env.observation_space,
            self.env.reward_range,
        )

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="ansi"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)


def main():
    manager = multiprocessing.Manager()
    return_queue = manager.Queue()
    judge_proc = Process(target=run_judge, args=(return_queue,))
    judge_proc.start()
    port = None
    for _ in range(10):  # wait for up to 10 seconds
        time.sleep(1)
        if not return_queue.empty():
            port = return_queue.get()
            break
    if not isinstance(port, int):
        raise Exception("judge process not properly initialized")
    try:
        n_runs = 1
        env = TaxiAgentEnv(port=port)
        evaluator = StepCountEvaluator()
        seeds = [2333 for _ in range(n_runs)]
        test_case = TaxiTestCase(
            t_max=10000,
            env=env,
            evaluator=evaluator,
            agent_init={},
            seeds=seeds,
            case_id=0,
            time_limit=3600,
            n_runs=n_runs,
        )
        test_suite = TestSuite(suite_id="elevator_test", cases=[test_case])
        res = test_suite.run(env.create_agent)
        print(res)
    finally:
        judge_proc.terminate()


if __name__ == "__main__":
    main()
