from box import Box
import argparse
import os
import json
import re
import shutil
from pathlib import Path

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
        "--data_path",
        type=str,
        default="",
        help="Override the dataset root, e.g. a cleaned USPTO_31k_UNMATCHED folder.",
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

    smiles_pattern = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|Se?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    def tokenize_smiles_line(line: str) -> str:
        line = line.strip()
        if not line:
            return line
        tokens = smiles_pattern.findall(line)
        if "".join(tokens) != line:
            return line
        return " ".join(tokens)

    def prepare_tokenized_dataset(src_root: str) -> str:
        src_root_path = Path(src_root)
        tokenized_root = Path("/kaggle/working") / f"{src_root_path.name}_tokenized"
        if tokenized_root.exists():
            shutil.rmtree(tokenized_root)
        tokenized_root.mkdir(parents=True, exist_ok=True)

        for filename in ["vocab_smiles.txt", "src-train.txt", "tgt-train.txt", "src-val.txt", "tgt-val.txt", "src-test.txt", "tgt-test.txt"]:
            source_file = src_root_path / filename
            target_file = tokenized_root / filename
            if filename == "vocab_smiles.txt":
                shutil.copyfile(source_file, target_file)
                continue
            with open(source_file, "r") as fr, open(target_file, "w") as fw:
                for line in fr:
                    fw.write(tokenize_smiles_line(line) + "\n")
        return str(tokenized_root)

    with open(args.config_json, "r") as fr:
        config_dict = json.load(fr)

    config_dict.setdefault("model", {})
    config_dict.setdefault("training", {})
    config_dict.setdefault("others", {})

    if args.save_dir:
        config_dict["model"]["save_dir"] = args.save_dir
    if args.data_path:
        config_dict["data"]["data_path"] = args.data_path
    if args.checkpoint_path:
        config_dict["training"]["checkpoint_path"] = args.checkpoint_path
        config_dict["training"]["resume_training"] = args.resume_training
        config_dict["model"]["pretrained_model_path"] = ""
    elif args.pretrained_model_path:
        ckpt_file = os.path.join(args.pretrained_model_path, "model", "valid_checkpoint.pt")
        if os.path.isfile(ckpt_file):
            config_dict["training"]["checkpoint_path"] = ckpt_file
            config_dict["training"]["resume_training"] = args.resume_training
            config_dict["model"]["pretrained_model_path"] = ""
        else:
            config_dict["model"]["pretrained_model_path"] = args.pretrained_model_path

    if "data" in config_dict and "data_path" in config_dict["data"]:
        config_dict["data"]["data_path"] = prepare_tokenized_dataset(config_dict["data"]["data_path"])

    config = Box(config_dict)
    config.others.local_rank = args.local_rank

    trainer = SequenceTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
