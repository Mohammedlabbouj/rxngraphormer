import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import List


SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Se?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def strip_all_whitespace(value: str) -> str:
    return "".join((value or "").split())


def tokenize_smiles_line(line: str) -> str:
    line = strip_all_whitespace(line)
    if not line:
        return line
    tokens = SMILES_PATTERN.findall(line)
    if "".join(tokens) != line:
        raise ValueError(f"Unable to tokenize SMILES: {line}")
    return " ".join(tokens)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def infer_input_format(data_config: dict, input_root: Path) -> str:
    requested_format = data_config.get("input_format", "auto").lower()
    if requested_format in {"csv", "paired_txt"}:
        return requested_format
    if (input_root / data_config.get("train_csv_file", "train.csv")).is_file():
        return "csv"
    if (input_root / data_config["train_src_file"]).is_file():
        return "paired_txt"
    raise FileNotFoundError(
        f"Could not infer fine-tuning input format under {input_root}. "
        "Expected either split CSV files or src/tgt text files."
    )


def copy_vocab_file(data_config: dict, input_root: Path, output_root: Path) -> None:
    vocab_name = data_config["vocab_file"]
    vocab_source_path = input_root / vocab_name
    if not vocab_source_path.is_file():
        vocab_source = data_config.get("vocab_source_path")
        if vocab_source:
            vocab_source_path = resolve_path(vocab_source)
    if not vocab_source_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_source_path}")
    shutil.copyfile(vocab_source_path, output_root / vocab_name)


def write_tokenized_txt_split(input_root: Path, output_root: Path, src_name: str, tgt_name: str) -> None:
    source_file = input_root / src_name
    target_file = input_root / tgt_name
    output_src = output_root / src_name
    output_tgt = output_root / tgt_name

    with open(source_file, "r") as fr, open(output_src, "w") as fw:
        for line in fr:
            fw.write(tokenize_smiles_line(line) + "\n")
    with open(target_file, "r") as fr, open(output_tgt, "w") as fw:
        for line in fr:
            fw.write(tokenize_smiles_line(line) + "\n")


def build_source_smiles(row: dict, source_columns: List[str]) -> str:
    parts = [strip_all_whitespace(row.get(column, "")) for column in source_columns]
    parts = [part for part in parts if part]
    return ".".join(parts)


def write_tokenized_csv_split(
    input_root: Path,
    output_root: Path,
    csv_name: str,
    src_name: str,
    tgt_name: str,
    source_columns: List[str],
    target_column: str,
    delimiter: str,
) -> None:
    input_csv = input_root / csv_name
    output_src = output_root / src_name
    output_tgt = output_root / tgt_name

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fr:
        reader = csv.DictReader(fr, delimiter=delimiter)
        missing_columns = [column for column in [*source_columns, target_column] if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise KeyError(f"Missing required CSV columns in {input_csv}: {missing_columns}")

        src_rows = []
        tgt_rows = []
        for row_idx, row in enumerate(reader, start=2):
            src_smiles = build_source_smiles(row, source_columns)
            tgt_smiles = strip_all_whitespace(row.get(target_column, ""))
            if not src_smiles or not tgt_smiles:
                raise ValueError(
                    f"Empty source or target in {input_csv} at CSV row {row_idx}. "
                    f"Source columns: {source_columns}, target column: {target_column}"
                )
            src_rows.append(tokenize_smiles_line(src_smiles))
            tgt_rows.append(tokenize_smiles_line(tgt_smiles))

    with open(output_src, "w") as fsrc:
        fsrc.write("\n".join(src_rows) + ("\n" if src_rows else ""))
    with open(output_tgt, "w") as ftgt:
        ftgt.write("\n".join(tgt_rows) + ("\n" if tgt_rows else ""))


def prepare_tokenized_dataset(data_config: dict) -> str:
    input_root = resolve_path(data_config["data_path"])
    output_root = input_root / "_rxngraphormer_tokenized"
    input_format = infer_input_format(data_config, input_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copy_vocab_file(data_config, input_root, output_root)

    if input_format == "csv":
        source_columns = data_config.get("csv_source_columns", ["Reactants", "Reagents"])
        target_column = data_config.get("csv_target_column", "Products")
        delimiter = data_config.get("csv_delimiter", ",")
        split_configs = [
            (data_config["train_csv_file"], data_config["train_src_file"], data_config["train_tgt_file"]),
            (data_config["valid_csv_file"], data_config["valid_src_file"], data_config["valid_tgt_file"]),
            (data_config["test_csv_file"], data_config["test_src_file"], data_config["test_tgt_file"]),
        ]
        for csv_name, src_name, tgt_name in split_configs:
            write_tokenized_csv_split(
                input_root=input_root,
                output_root=output_root,
                csv_name=csv_name,
                src_name=src_name,
                tgt_name=tgt_name,
                source_columns=source_columns,
                target_column=target_column,
                delimiter=delimiter,
            )
    else:
        split_configs = [
            (data_config["train_src_file"], data_config["train_tgt_file"]),
            (data_config["valid_src_file"], data_config["valid_tgt_file"]),
            (data_config["test_src_file"], data_config["test_tgt_file"]),
        ]
        for src_name, tgt_name in split_configs:
            write_tokenized_txt_split(
                input_root=input_root,
                output_root=output_root,
                src_name=src_name,
                tgt_name=tgt_name,
            )

    return str(output_root)


def main():
    from box import Box

    from rxngraphormer.train import SequenceTrainer

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
        config_dict["data"]["data_path"] = prepare_tokenized_dataset(config_dict["data"])

    config = Box(config_dict)
    config.others.local_rank = args.local_rank

    trainer = SequenceTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
