from .my_agent import DQNAgent


def create_agent(test_case_id, *args, **kwargs):
    return DQNAgent()
