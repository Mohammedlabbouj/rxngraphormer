from box import Box
import argparse
import json

from rxngraphormer.train import SequenceTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_json",
        type=str,
        default="./config/uspto_31k_unmatched_finetune.json",
        help="Base fine-tuning config for USPTO_31k_UNMATCHED.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="Override the pretrained USPTO_480k model folder path, e.g. /kaggle/input/.../USPTO_480k",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Override the output root directory. Defaults to the config value.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional checkpoint to load before fine-tuning.",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume optimizer and scheduler state from --checkpoint_path.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config_json, "r") as fr:
        config_dict = json.load(fr)

    config_dict.setdefault("model", {})
    config_dict.setdefault("training", {})
    config_dict.setdefault("others", {})

    if args.pretrained_model_path:
        config_dict["model"]["pretrained_model_path"] = args.pretrained_model_path
    if args.save_dir:
        config_dict["model"]["save_dir"] = args.save_dir
    if args.checkpoint_path:
        config_dict["training"]["checkpoint_path"] = args.checkpoint_path
        config_dict["training"]["resume_training"] = args.resume_training

    config = Box(config_dict)
    config.others.local_rank = args.local_rank

    trainer = SequenceTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
