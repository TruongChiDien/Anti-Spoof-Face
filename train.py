import argparse
from src.train_main import TrainMain
from src.default_config import update_config, save_config


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
    save_config(conf, 'configs/test_config.yaml')
    trainer = TrainMain(conf)
    trainer.train_model()