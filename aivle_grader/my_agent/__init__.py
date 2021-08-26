from .my_agent import DQNAgent


def create_agent(case_id, *args, **kwargs):
    return DQNAgent(*args, **kwargs)
