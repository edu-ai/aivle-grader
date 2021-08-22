from abc import ABCMeta
from typing import List, Callable

from aivle_grader.abc.agent import Agent
from aivle_grader.abc.test_case import TestCase


class TestSuite(metaclass=ABCMeta):
    def __init__(self, suite_id, cases: List[TestCase]):
        self.suite_id = suite_id
        self.test_cases = cases

    def run(self, create_agent: Callable[[], Agent]):
        results = []
        for case in self.test_cases:
            res = case.evaluate(create_agent)
            results.append(res)
        return results
