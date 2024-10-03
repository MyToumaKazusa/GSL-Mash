import dotenv
import hydra
from omegaconf import DictConfig
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


os.environ["WANDB_API_KEY"] = "416a0be766ec4c4b1323e6be331d4d6255f17e6f"
os.environ["WANDB_MODE"] = "offline"

@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
