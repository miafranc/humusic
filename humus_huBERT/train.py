# train.py

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from dataloader import LyricsDataset, BinaryLyricsDataset

# Check if tensorboard is available
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not installed. Logging will be disabled.")


# Global variable to store class names for logging
_class_names = None


def set_class_names(class_names):
    """Set class names for per-class metric logging."""
    global _class_names
    _class_names = class_names


def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.5).astype(int)
    
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Compute per-class F1 scores
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
    }
    
    # Add per-class F1 scores to metrics
    global _class_names
    if (_class_names is not None):
        for i, class_name in enumerate(_class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    else:
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
    
    return metrics


def compute_binary_metrics(eval_pred):
    """Compute metrics for binary classification (one-vs-one)."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary', zero_division=0),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
    }


def log_per_class_f1(results, class_names=None):
    """
    Log per-class F1 scores in a readable format.
    
    Args:
        results: Dictionary with evaluation results
        class_names: List of class names
    """
    print(f"\n{'='*50}")
    print("Per-Class F1 Scores")
    print(f"{'='*50}")
    print(f"{'Class':<25} {'F1 Score':>15}")
    print(f"{'-'*50}")
    
    f1_scores = []
    if class_names is not None:
        for class_name in class_names:
            key = f'eval_f1_{class_name}'
            if key in results:
                f1 = results[key]
                f1_scores.append((class_name, f1))
                print(f"{class_name:<25} {f1:>15.4f}")
    else:
        # Fallback to numeric class indices
        for key, value in sorted(results.items()):
            if key.startswith('eval_f1_class_') or key.startswith('eval_f1_'):
                class_name = key.replace('eval_f1_', '')
                f1_scores.append((class_name, value))
                print(f"{class_name:<25} {value:>15.4f}")
    
    print(f"{'-'*50}")
    print(f"{'Micro F1':<25} {results.get('eval_f1_micro', 0):>15.4f}")
    print(f"{'Macro F1':<25} {results.get('eval_f1_macro', 0):>15.4f}")
    print(f"{'='*50}\n")


def load_model_and_tokenizer(model_name, num_labels):
    """Load pretrained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # Move model to available GPU (respects CUDA_VISIBLE_DEVICES set by SLURM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    return model, tokenizer


def create_datasets(X_train, y_train, X_val, y_val, tokenizer, max_length=512):
    """Create train and validation datasets."""
    train_dataset = LyricsDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = LyricsDataset(X_val, y_val, tokenizer, max_length)
    return train_dataset, val_dataset


def get_training_args(output_dir, epochs=5, batch_size=8, learning_rate=2e-5):
    """Get training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        save_total_limit=1,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        report_to="tensorboard" if TENSORBOARD_AVAILABLE else "none"
    )


def train_model(
    X_train, y_train, X_val, y_val,
    num_labels,
    output_dir,
    model_name="SZTAKI-HLT/hubert-base-cc",
    max_length=512,
    epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    class_names=None
):
    """Train the multi-label classification model."""
    
    # Set class names for per-class metrics
    if class_names is not None:
        set_class_names(class_names)
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)
    
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        X_train, y_train, X_val, y_val, tokenizer, max_length
    )
    
    training_args = get_training_args(output_dir, epochs, batch_size, learning_rate)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Final evaluation:")
    results = trainer.evaluate()
    print(results)
    
    # Log per-class F1 scores
    log_per_class_f1(results, class_names)
    
    return trainer, results


def train_binary_model(
    X_train, y_train, X_val, y_val,
    output_dir,
    model_name="SZTAKI-HLT/hubert-base-cc",
    max_length=512,
    epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    genre_a=None,
    genre_b=None,
):
    """Train a binary classification model for one-vs-one genre pair."""
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    print("Creating datasets...")
    train_dataset = BinaryLyricsDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = BinaryLyricsDataset(X_val, y_val, tokenizer, max_length)
    
    training_args = get_training_args(output_dir, epochs, batch_size, learning_rate)
    # Override metric for best model
    training_args.metric_for_best_model = "f1"
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_binary_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save genre pair info
    pair_info = {"genre_a": genre_a, "genre_b": genre_b}
    import json
    with open(f"{output_dir}/genre_pair.json", "w") as f:
        json.dump(pair_info, f, ensure_ascii=False, indent=2)
    
    print("Final evaluation:")
    results = trainer.evaluate()
    print(results)
    
    return trainer, results
