import argparse
import re
import shutil
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Se?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def load_vocab(vocab_file: str):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            token = token.split()[0]
            vocab[token] = i
    return vocab


def smi_tokenizer(smi: str) -> str:
    tokens = [token for token in TOKEN_PATTERN.findall(smi)]
    if smi != "".join(tokens):
        raise ValueError(f"Unable to tokenize SMILES: {smi}")
    return " ".join(tokens)


def normalize_smiles_block(smi: str) -> str:
    smi = smi.strip()
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def normalize_line(line: str, canonicalize: bool) -> str:
    line = line.strip()
    if not line:
        return ""
    if canonicalize:
        parts = line.split(".")
        canon_parts = []
        for part in parts:
            canon = normalize_smiles_block(part)
            if not canon:
                return ""
            canon_parts.append(canon)
        line = ".".join(canon_parts)

    if " " in line:
        return " ".join(line.split())

    tokens = TOKEN_PATTERN.findall(line)
    if "".join(tokens) != line:
        return ""
    return smi_tokenizer(line)


def load_split(path: Path):
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="dataset/USPTO_31k_UNMATCHED",
        help="Raw USPTO_31k_UNMATCHED folder containing src/tgt txt files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset/USPTO_31k_UNMATCHED_clean",
        help="Output folder for the cleaned tokenized dataset.",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize each molecule with RDKit before tokenization.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(str(input_root / "vocab_smiles.txt"))
    vocab_missing = []

    shutil.copyfile(input_root / "vocab_smiles.txt", output_root / "vocab_smiles.txt")

    splits = ["train", "val", "test"]
    total_in = 0
    total_out = 0
    total_dropped = 0

    for split in splits:
        src_in = load_split(input_root / f"src-{split}.txt")
        tgt_in = load_split(input_root / f"tgt-{split}.txt")
        if len(src_in) != len(tgt_in):
            raise ValueError(f"Length mismatch for {split}: {len(src_in)} vs {len(tgt_in)}")

        out_src = []
        out_tgt = []
        dropped_examples = []

        for idx, (src_line, tgt_line) in enumerate(tqdm(list(zip(src_in, tgt_in)), desc=f"Cleaning {split}", total=len(src_in))):
            total_in += 1
            src_clean = normalize_line(src_line, args.canonicalize)
            tgt_clean = normalize_line(tgt_line, args.canonicalize)

            if not src_clean or not tgt_clean:
                total_dropped += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append((idx, src_line, tgt_line, "invalid_smiles_or_tokenization"))
                continue

            src_tokens = src_clean.split()
            tgt_tokens = tgt_clean.split()
            missing = [tok for tok in src_tokens + tgt_tokens if tok not in vocab]

            if missing:
                total_dropped += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append((idx, src_line, tgt_line, f"oov={sorted(set(missing))[:5]}"))
                continue

            out_src.append(src_clean)
            out_tgt.append(tgt_clean)
            total_out += 1

        with open(output_root / f"src-{split}.txt", "w") as fsrc, open(output_root / f"tgt-{split}.txt", "w") as ftgt:
            fsrc.write("\n".join(out_src) + ("\n" if out_src else ""))
            ftgt.write("\n".join(out_tgt) + ("\n" if out_tgt else ""))

        print(f"[INFO] {split}: kept {len(out_src)} / {len(src_in)}")
        if dropped_examples:
            print(f"[INFO] Sample dropped rows for {split}:")
            for row in dropped_examples[:5]:
                print(f"  idx={row[0]} reason={row[3]}")
                print(f"    src={row[1]}")
                print(f"    tgt={row[2]}")

    print(f"[INFO] Total input rows: {total_in}")
    print(f"[INFO] Total kept rows: {total_out}")
    print(f"[INFO] Total dropped rows: {total_dropped}")
    print(f"[INFO] Cleaned dataset written to: {output_root}")


if __name__ == "__main__":
    main()
