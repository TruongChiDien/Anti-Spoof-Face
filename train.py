import argparse
from src.train_main import TrainMain
from src.default_config import get_default_config, update_config


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = update_config(args)
    trainer = TrainMain(conf)
    trainer.train_model()

