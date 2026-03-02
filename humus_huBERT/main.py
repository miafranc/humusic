# main.py

import os
import argparse
import json
import torch
import warnings
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold

from dataloader import (
    load_data,
    prepare_labels,
    split_data,
    save_label_mapping,
    load_label_mapping,
    stratified_subset,
    stratified_multilabel_kfold,
    log_class_distribution,
    get_genre_pairs,
    filter_data_for_pair,
)
from train import train_model, train_binary_model

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
FAST_TRAIN_SUBSET_SIZE = 0.1  # Use 10% of data for fast training
N_FOLDS = 5  # Number of folds for cross-validation

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device():
    """Setup and print device information."""
    if torch.cuda.is_available():
        # SLURM sets CUDA_VISIBLE_DEVICES, PyTorch will see only the assigned GPU(s)
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")


def run_training(json_path, output_dir, model_name=DEFAULT_MODEL_NAME, fast_train=False, cross_validate=False):
    """Run the full training pipeline."""
    
    # Load data
    print("Loading data...")
    lyrics, tags = load_data(json_path)
    print(f"Loaded {len(lyrics)} songs")
    
    # Encode labels
    encoded_labels, mlb = prepare_labels(tags)
    num_labels = len(mlb.classes_)
    print(f"Found {num_labels} unique genres: {list(mlb.classes_)}")
    
    # Create stratified subset for fast training
    if fast_train:
        print(f"Fast train mode: using {FAST_TRAIN_SUBSET_SIZE*100:.0f}% of data...")
        lyrics, encoded_labels = stratified_subset(
            lyrics, encoded_labels, 
            subset_size=FAST_TRAIN_SUBSET_SIZE
        )
        print(f"Subset size: {len(lyrics)} songs")
    
    # Save label mapping
    os.makedirs(output_dir, exist_ok=True)
    save_label_mapping(mlb, f'{output_dir}/label_mapping.json')
    
    # Reduce epochs for fast training
    epochs = 3 if fast_train else EPOCHS
    
    if cross_validate:
        # Cross-validation training
        print(f"Running {N_FOLDS}-fold cross-validation...")
        splits = stratified_multilabel_kfold(lyrics, encoded_labels, n_splits=N_FOLDS)
        
        all_results = []
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{N_FOLDS}")
            print(f"{'='*50}")
            
            # Create train/val sets for this fold
            X_train = [lyrics[i] for i in train_indices]
            X_val = [lyrics[i] for i in val_indices]
            y_train = encoded_labels[train_indices]
            y_val = encoded_labels[val_indices]
            
            print(f"Train: {len(X_train)}, Validation: {len(X_val)}")

            log_class_distribution(y_train, y_val, mlb, fold_num=fold_idx + 1)
            
            fold_output_dir = f"{output_dir}/fold_{fold_idx + 1}"
            
            trainer, result = train_model(
                X_train, y_train, X_val, y_val,
                num_labels=num_labels,
                output_dir=fold_output_dir,
                model_name=model_name,
                max_length=MAX_LENGTH,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE
            )
            
            if isinstance(result, tuple):
                trainer = result[0]
        # Print cross-validation summary
        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        f1_micros = [r['eval_f1_micro'] for r in all_results]
        f1_macros = [r['eval_f1_macro'] for r in all_results]
        print(f"F1 Micro: {np.mean(f1_micros):.4f} (+/- {np.std(f1_micros):.4f})")
        print(f"F1 Macro: {np.mean(f1_macros):.4f} (+/- {np.std(f1_macros):.4f})")
        
        return None
    else:
        # Single train/val split
        X_train, X_val, y_train, y_val = split_data(lyrics, encoded_labels)
        print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        
        trainer = train_model(
            X_train, y_train, X_val, y_val,
            num_labels=num_labels,
            output_dir=output_dir,
            model_name=model_name,
            max_length=MAX_LENGTH,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        return trainer


def run_ovo_training(json_path, output_dir, model_name=DEFAULT_MODEL_NAME, fast_train=False, n_folds=N_FOLDS):
    """Run one-vs-one binary classification for all genre pairs with k-fold CV."""
    
    # Load raw data (we need original tags, not binarized)
    print("Loading data...")
    lyrics, tags = load_data(json_path)
    print(f"Loaded {len(lyrics)} songs")
    
    # Get all genre pairs
    genre_pairs = get_genre_pairs(tags)
    print(f"Total genre pairs to evaluate: {len(genre_pairs)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = 3 if fast_train else EPOCHS
    
    all_pair_results = {}
    
    for pair_idx, (genre_a, genre_b) in enumerate(genre_pairs):
        print(f"\n{'#'*60}")
        if pair_idx < 41:
            continue

        print(f"Pair {pair_idx + 1}/{len(genre_pairs)}: {genre_a} vs {genre_b}")
        print(f"{'#'*60}")
        
        # Filter data for this pair
        pair_lyrics, pair_labels = filter_data_for_pair(lyrics, tags, genre_a, genre_b)
        
        if len(pair_lyrics) < n_folds * 2:
            print(f"  Skipping: only {len(pair_lyrics)} samples (need at least {n_folds * 2})")
            continue
        
        count_a = int(np.sum(pair_labels == 0))
        count_b = int(np.sum(pair_labels == 1))
        print(f"  {genre_a}: {count_a} songs, {genre_b}: {count_b} songs, total: {len(pair_lyrics)}")
        
        # Optionally subsample for fast training
        if fast_train:
            subset_size = max(int(len(pair_lyrics) * FAST_TRAIN_SUBSET_SIZE), n_folds * 2)
            if subset_size < len(pair_lyrics):
                indices = np.arange(len(pair_lyrics))
                np.random.shuffle(indices)
                indices = indices[:subset_size]
                pair_lyrics = [pair_lyrics[i] for i in indices]
                pair_labels = pair_labels[indices]
                print(f"  Fast train: using {len(pair_lyrics)} samples")
        
        # K-fold cross-validation
        pair_lyrics_array = np.array(pair_lyrics)
        pair_labels_array = np.array(pair_labels)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pair_lyrics_array, pair_labels_array)):
            print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")
            
            X_train = pair_lyrics_array[train_idx].tolist()
            X_val = pair_lyrics_array[val_idx].tolist()
            y_train = pair_labels_array[train_idx]
            y_val = pair_labels_array[val_idx]
            
            print(f"  Train: {len(X_train)} ({np.sum(y_train==0)} {genre_a}, {np.sum(y_train==1)} {genre_b})")
            print(f"  Val:   {len(X_val)} ({np.sum(y_val==0)} {genre_a}, {np.sum(y_val==1)} {genre_b})")
            
            fold_output_dir = f"{output_dir}/ovo_{genre_a}_vs_{genre_b}/fold_{fold_idx + 1}"
            
            trainer, result = train_binary_model(
                X_train, y_train, X_val, y_val,
                output_dir=fold_output_dir,
                model_name=model_name,
                max_length=MAX_LENGTH,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                genre_a=genre_a,
                genre_b=genre_b,
            )
            
            fold_results.append(result)
        
        # Summarize this pair
        pair_key = f"{genre_a}_vs_{genre_b}"
        f1s = [r['eval_f1'] for r in fold_results]
        accs = [r['eval_accuracy'] for r in fold_results]
        
        all_pair_results[pair_key] = {
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'n_samples': len(pair_lyrics),
            'count_a': count_a,
            'count_b': count_b,
        }
        
        print(f"\n  {genre_a} vs {genre_b}: F1 = {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f}), "
              f"Acc = {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("One-vs-One Classification Summary")
    print(f"{'='*70}")
    print(f"{'Pair':<30} {'F1':>12} {'Accuracy':>12} {'Samples':>10}")
    print(f"{'-'*70}")
    for pair_key, res in sorted(all_pair_results.items(), key=lambda x: x[1]['f1_mean'], reverse=True):
        print(f"{pair_key:<30} {res['f1_mean']:.4f}±{res['f1_std']:.4f} "
              f"{res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f} {res['n_samples']:>10}")
    print(f"{'='*70}")
    
    # Save summary
    summary_path = f"{output_dir}/ovo_summary.json"
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_pair_results, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")
    
    return all_pair_results


def predict(text, model_path, threshold=0.5):
    """Predict genres for new lyrics."""
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load label mapping
    label_mapping = load_label_mapping(f'{model_path}/label_mapping.json')
    
    # Preprocess
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    
    # Get predicted genres
    predicted_genres = []
    for i, prob in enumerate(probs):
        if prob > threshold:
            predicted_genres.append({
                'genre': label_mapping[str(i)],
                'confidence': float(prob)
            })
    
    return sorted(predicted_genres, key=lambda x: x['confidence'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description='Lyrics Genre Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--json_path', type=str, required=True, help='Path to JSON file'
    )
    train_parser.add_argument(
        '--output_dir', type=str, default='./lyrics_classifier', help='Output directory'
    )
    train_parser.add_argument(
        '--model_name', type=str, default=DEFAULT_MODEL_NAME, help='Pretrained model name'
    )
    train_parser.add_argument(
        '--fast_train', action='store_true', 
        help='Run on small stratified subset for quick error checking'
    )
    train_parser.add_argument(
        '--cross_validate', action='store_true',
        help=f'Run {N_FOLDS}-fold cross-validation instead of single split'
    )
    
    # Train OVO command
    ovo_parser = subparsers.add_parser('train_ovo', help='Train one-vs-one binary classifiers for all genre pairs')
    ovo_parser.add_argument(
        '--json_path', type=str, required=True, help='Path to JSON file'
    )
    ovo_parser.add_argument(
        '--output_dir', type=str, default='./lyrics_classifier_ovo', help='Output directory'
    )
    ovo_parser.add_argument(
        '--model_name', type=str, default=DEFAULT_MODEL_NAME, help='Pretrained model name'
    )
    ovo_parser.add_argument(
        '--fast_train', action='store_true',
        help='Run on small subset for quick error checking'
    )
    ovo_parser.add_argument(
        '--n_folds', type=int, default=N_FOLDS,
        help=f'Number of folds for cross-validation (default: {N_FOLDS})'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict genres')
    predict_parser.add_argument(
        '--text', type=str, required=True, help='Lyrics text to classify'
    )
    predict_parser.add_argument(
        '--model_path', type=str, default='./lyrics_classifier', help='Path to trained model'
    )
    predict_parser.add_argument(
        '--threshold', type=float, default=0.5, help='Prediction threshold'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        setup_device()  # Print device info before training
        os.makedirs(args.output_dir, exist_ok=True)
        run_training(args.json_path, args.output_dir, args.model_name, args.fast_train, args.cross_validate)
        
    elif args.command == 'train_ovo':
        setup_device()
        os.makedirs(args.output_dir, exist_ok=True)
        run_ovo_training(args.json_path, args.output_dir, args.model_name, args.fast_train, args.n_folds)
        
    elif args.command == 'predict':
        setup_device()  # Print device info before prediction
        genres = predict(args.text, args.model_path, args.threshold)
        print("Predicted genres:")
        if genres:
            for g in genres:
                print(f"  {g['genre']}: {g['confidence']:.3f}")
        else:
            print("  No genres predicted above threshold")
    else:
        parser.print_help()



if __name__ == "__main__":
    set_seed(42)
    main()