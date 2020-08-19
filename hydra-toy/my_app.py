import hydra
from omegaconf import DictConfig
# Note: omegaconf info: https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation
# https://mybinder.org/v2/gh/omry/omegaconf/master?filepath=docs%2Fnotebook%2FTutorial.ipynb

@hydra.main(config_path="config.yaml")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()