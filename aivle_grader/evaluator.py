from aivle_grader.abc.evaluator import Evaluator, EvaluationResult


class RewardEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.error = None
        self.episodes = []
        self.total_reward = 0

    def reset(self) -> None:
        self.episodes = []

    def run(self) -> None:
        self.episodes.append([])

    def step(self, full_state) -> None:
        self.episodes[-1].append(full_state["reward"])

    def done(self) -> None:
        # TODO
        total_reward_per_episode = [sum(r) for r in self.episodes]
        self.total_reward = sum(total_reward_per_episode)

    def get_result(self) -> EvaluationResult:
        return EvaluationResult(
            name="reward evaluation",
            value=self.total_reward / len(self.episodes),
            results=self.episodes,
            error=self.error,
        )
