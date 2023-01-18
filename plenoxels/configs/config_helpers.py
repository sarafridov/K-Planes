import argparse
import os
from datetime import datetime


def parse_config(defaults):
    # Build experiment configuration
    parser = argparse.ArgumentParser("Train + evaluate kernel model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-updates", default=[], nargs='*')
    parser.add_argument("--logdir", default=None)
    args = parser.parse_args()
    # Allow up to two configs, one for reloading and one for training
    reload_cfg = None
    #train_cfg = defaults
    train_cfg = None
    # assert args.logdir is not None or args.config is not None, "Must specify at least one config"
    # Passing both a logdir and a config means train new scenes using pretrained dicts
    if args.logdir is not None:
        logged_config_file = os.path.join(args.logdir, "config.yaml")
        if not os.path.isfile(logged_config_file):
            raise RuntimeError(f"logdir {args.logdir} doesn't specify a config-file")
        print(f"Loading configuration from logs at {logged_config_file}")
        reload_cfg = defaults.clone()
        reload_cfg.merge_from_file(logged_config_file)
        reload_cfg.merge_from_list(args.config_updates)
    if args.config is not None:
        train_cfg = defaults.clone()
        # Reuse the same config as was reloaded, but make updates for datasets and logdir
        if reload_cfg is not None:
            train_cfg.merge_from_file(logged_config_file)
        train_cfg.merge_from_file(args.config)
        train_cfg.merge_from_list(args.config_updates)
    print(f"[{datetime.now()}] Starting")
    return train_cfg, reload_cfg

