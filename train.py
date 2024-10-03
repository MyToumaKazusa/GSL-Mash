import dotenv
import hydra
from omegaconf import DictConfig
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

os.environ["WANDB_API_KEY"] = "416a0be766ec4c4b1323e6be331d4d6255f17e6f"
os.environ["WANDB_MODE"] = "offline"


@hydra.main(config_path="configs/", config_name="train.yaml",version_base= "1.2") #随着版本更新，这里要指定version=1.2，否则会导致hydra无法实例化类
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)
    
    # Train model
    return train(config)


if __name__ == "__main__":
    main()
