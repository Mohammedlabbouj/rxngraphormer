#!/usr/bin/env python
"""
Script to test RXNGraphormer model on custom data and store top-k accuracies.

This script loads a trained model, reads test reactions from the dataset folder,
evaluates the model, and stores top-k accuracy results for each reaction in a CSV file.

Usage:
    python test_model_topk.py --model_path <path_to_model> --data_path <path_to_dataset> --output_file <output_csv>
    
Example:
    python test_model_topk.py \
        --model_path /home/mohammed.labbouj/lustre/isti_ai-86tuechqypa/users/mohammed.labbouj/model_benmark/RXNGraphormer/model_path/USPTO_480k \
        --data_path /home/mohammed.labbouj/lustre/isti_ai-86tuechqypa/users/mohammed.labbouj/model_benmark/RXNGraphormer/dataset/USPTO_480k \
        --output_file topk_results.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from box import Box
from rdkit import Chem
from torch_geometric.loader import DataLoader

# Add the workspace to path to import rxngraphormer
sys.path.insert(0, '/workspace')

from rxngraphormer.model import RXNGraphormer
from rxngraphormer.data import load_vocab, RXNG2SDataset, smi_tokenizer
from rxngraphormer.utils import update_dict_key

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, ckpt_file="valid_checkpoint.pt"):
    """Load the trained model from the specified path."""
    print(f"[INFO] Loading trained model from {model_path}")
    
    # Load parameters
    trained_para_json = f"{model_path}/parameters.json"
    with open(trained_para_json, 'r') as fr:
        pretrained_config_dict = json.load(fr)
    trained_config = Box(pretrained_config_dict)
    
    # Load vocabulary
    vocab = load_vocab(f'{trained_config.data.data_path}/{trained_config.data.vocab_file}')
    vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
    
    # Create model
    rxng = RXNGraphormer("sequence_generation", trained_config, vocab)
    model = rxng.get_model()
    
    # Load checkpoint
    ckpt_path = f"{model_path}/model/{ckpt_file}"
    ckpt_inf = torch.load(ckpt_path, map_location=device)
    
    model.to(device)
    model.load_state_dict(update_dict_key(ckpt_inf['model_state_dict']))
    model.eval()
    
    print(f"[INFO] Model loaded successfully!")
    return model, trained_config, vocab_rev


def load_test_data(data_path, config, vocab_file="vocab_smiles.txt", src_file="src-test.txt", tgt_file="tgt-test.txt"):
    """Load test dataset from the specified path."""
    print(f"[INFO] Loading test dataset from {data_path}")
    
    test_dataset = RXNG2SDataset(
        root=data_path,
        src_file=src_file,
        tgt_file=tgt_file,
        vocab_file=vocab_file,
        trunck=0,
        multi_process=False,
        oh=False
    )
    
    # Extract ground truth SMILES
    vocab_rev = [k for k, v in sorted(load_vocab(f'{data_path}/{vocab_file}').items(), key=lambda tup: tup[1])]
    ground_truth_smiles_lst = [
        "".join([vocab_rev[idx] for idx in test_dataset[idx].tgt_token_ids[0][:int(test_dataset[idx].tgt_lens[0])-1]])
        for idx in range(len(test_dataset))
    ]
    
    print(f"[INFO] Loaded {len(test_dataset)} test samples")
    return test_dataset, ground_truth_smiles_lst


def evaluate_model(model, dataloader, ground_truth_smiles_lst, vocab_rev, n_best=10, beam_size=10, 
                   temperature=1.0, min_length=1, max_length=512):
    """Evaluate the model and compute top-k accuracies for each sample."""
    model.eval()
    all_predictions = []
    
    print(f"[INFO] Running inference with beam_size={beam_size}, n_best={n_best}, temperature={temperature}")
    
    with torch.no_grad():
        step = 0
        for batch_data in dataloader:
            step += 1
            if step % 10 == 0:
                print(f"[INFO] Processing batch {step}")
            
            batch_data = batch_data.to(device)
            results = model.infer(
                reaction_batch=batch_data,
                batch_size=len(batch_data.tgt_lens),
                beam_size=beam_size,
                n_best=n_best,
                temperature=temperature,
                min_length=min_length,
                max_length=max_length
            )
            
            for predictions in results["predictions"]:
                smis = []
                for prediction in predictions:
                    predicted_idx = prediction.detach().cpu().numpy()
                    predicted_tokens = [vocab_rev[idx] for idx in predicted_idx[:-1]]
                    smi = "".join(predicted_tokens)
                    smis.append(smi)
                all_predictions.append(smis)
    
    print(f"[INFO] Inference complete. Computing accuracies...")
    
    # Compute accuracies for each sample
    num_samples = len(ground_truth_smiles_lst)
    num_predictions = len(all_predictions[0]) if all_predictions else 0
    
    # Store results: each row is a sample, columns are top-1, top-2, ..., top-n_best accuracy (1 if correct, 0 otherwise)
    accuracies = np.zeros([num_samples, num_predictions], dtype=np.float32)
    
    for i in range(num_samples):
        smi_tgt = ground_truth_smiles_lst[i]
        line_predict = all_predictions[i]
        
        # Canonicalize predictions
        smis_predict = []
        for smi in line_predict:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smi = Chem.MolToSmiles(mol)
                smis_predict.append(canonical_smi)
            else:
                smis_predict.append("")
        
        # Check if ground truth matches any of the top-k predictions
        # Canonicalize ground truth
        tgt_mol = Chem.MolFromSmiles(smi_tgt)
        canonical_tgt = Chem.MolToSmiles(tgt_mol) if tgt_mol else smi_tgt
        
        for j, smi in enumerate(smis_predict):
            if smi == canonical_tgt:
                # Mark this and all subsequent ranks as correct (1.0)
                accuracies[i, j:] = 1.0
                break
    
    return all_predictions, accuracies


def save_results(predictions, accuracies, ground_truth_smiles_lst, output_file, n_best):
    """Save results to CSV file."""
    print(f"[INFO] Saving results to {output_file}")
    
    # Create DataFrame with top-k predictions
    pred_columns = [f"Top-{i+1}_prediction" for i in range(len(predictions[0]))]
    pred_df = pd.DataFrame(predictions, columns=pred_columns)
    
    # Add ground truth
    pred_df['Ground_Truth'] = ground_truth_smiles_lst
    
    # Add top-k accuracy columns (cumulative: 1 if correct at this rank or before)
    acc_columns = [f"Top-{i+1}_correct" for i in range(n_best)]
    for i in range(min(n_best, accuracies.shape[1])):
        pred_df[acc_columns[i]] = accuracies[:, i]
    
    # Add overall metrics summary at the end
    summary_row = {'Ground_Truth': 'SUMMARY'}
    for i in range(min(n_best, accuracies.shape[1])):
        summary_row[acc_columns[i]] = np.mean(accuracies[:, i])
    pred_df.loc[len(pred_df)] = summary_row
    
    pred_df.to_csv(output_file, index=False)
    print(f"[INFO] Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("TOP-K ACCURACY SUMMARY")
    print("="*50)
    for i in range(min(n_best, accuracies.shape[1])):
        print(f"Top-{i+1} Accuracy: {np.mean(accuracies[:, i]):.4f} ({np.sum(accuracies[:, i])}/{len(accuracies[:, i])})")
    print("="*50)
    
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Test RXNGraphormer model and compute top-k accuracies")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory (containing parameters.json and model/)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset directory (containing src-test.txt, tgt-test.txt, vocab_smiles.txt)")
    parser.add_argument("--output_file", type=str, default="topk_results.csv",
                        help="Output CSV file to store results (default: topk_results.csv)")
    
    # Optional arguments
    parser.add_argument("--ckpt_file", type=str, default="valid_checkpoint.pt",
                        help="Checkpoint file name (default: valid_checkpoint.pt)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--beam_size", type=int, default=10,
                        help="Beam size for decoding (default: 10)")
    parser.add_argument("--n_best", type=int, default=10,
                        help="Number of best hypotheses to keep (default: 10)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--min_length", type=int, default=1,
                        help="Minimum sequence length (default: 1)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--src_file", type=str, default="src-test.txt",
                        help="Source test file name (default: src-test.txt)")
    parser.add_argument("--tgt_file", type=str, default="tgt-test.txt",
                        help="Target test file name (default: tgt-test.txt)")
    parser.add_argument("--vocab_file", type=str, default="vocab_smiles.txt",
                        help="Vocabulary file name (default: vocab_smiles.txt)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.data_path):
        print(f"[ERROR] Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Load model
    model, config, vocab_rev = load_model(args.model_path, ckpt_file=args.ckpt_file)
    
    # Load test data
    test_dataset, ground_truth_smiles_lst = load_test_data(
        args.data_path, 
        config,
        vocab_file=args.vocab_file,
        src_file=args.src_file,
        tgt_file=args.tgt_file
    )
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate model
    predictions, accuracies = evaluate_model(
        model, 
        test_dataloader, 
        ground_truth_smiles_lst, 
        vocab_rev,
        n_best=args.n_best,
        beam_size=args.beam_size,
        temperature=args.temperature,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Save results
    save_results(predictions, accuracies, ground_truth_smiles_lst, args.output_file, args.n_best)
    
    print(f"\n[SUCCESS] Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
