from bert_ordinal.scripts.train import TrainerAndEvaluator as TrainerAndEvaluatorBase
import wandb


class TrainerAndEvaluator(TrainerAndEvaluatorBase):
    def __init__(self):
        super().__init__()
        wandb.config.update(self.args)


def main():
    wandb.init(project="mt-ord-bert", entity="frobertson")
    TrainerAndEvaluator().train()


if __name__ == "__main__":
    main()
