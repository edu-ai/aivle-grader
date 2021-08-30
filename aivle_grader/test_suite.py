from typing import List, Callable

from aivle_grader.abc.agent import Agent
from aivle_grader.abc.evaluator import EvaluationResult
from aivle_grader.abc.test_case import TestCase


class TestResult:
    def __init__(self, case_id, result: EvaluationResult):
        self.case_id = case_id
        self.result = result

    def __str__(self):
        return f"<Case {self.case_id}> {self.result}"

    def __repr__(self):
        return str(self)


class TestSuite:
    """TestSuite is a collection of TestCase."""

    def __init__(self, suite_id, cases: List[TestCase]):
        self.suite_id = suite_id
        self.test_cases = cases

    def run(self, create_agent: Callable[..., Agent]) -> List[TestResult]:
        """Run the test cases in this test suite.

        :param create_agent: a function that returns an `Agent` object. Parameters
        to this function will be passed to the constructor of the underlying `Agent` object.
        :return:
        """
        results = []
        for case in self.test_cases:
            res = case.evaluate(create_agent)
            results.append(TestResult(case_id=case.case_id, result=res))
        return results
