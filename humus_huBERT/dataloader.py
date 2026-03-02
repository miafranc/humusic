# dataloader.py

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from itertools import combinations

# Genres to exclude from training
EXCLUDED_GENRES = {'country', 'reggae', 'rnb'}


def load_data(json_path):
    """Load JSON data and extract lyrics (with titles) and tags.
    
    Excludes songs that have any of the excluded genres (country, reggae, rnb).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lyrics = []
    tags = []
    excluded_count = 0
    
    for key, value in data.items():
        song_tags = set(value['tags'])
        
        # Skip songs with excluded genres
        if song_tags & EXCLUDED_GENRES:
            excluded_count += 1
            continue
        
        # Concatenate title with lyrics
        title = value.get('title', '')
        song_lyrics = value['lyrics']
        combined_text = f"{title} {song_lyrics}" if title else song_lyrics
        
        lyrics.append(combined_text)
        tags.append(value['tags'])
    
    print(f"Excluded {excluded_count} songs with genres: {EXCLUDED_GENRES}")
    
    return lyrics, tags


def prepare_labels(tags):
    """Encode multi-labels using MultiLabelBinarizer."""
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(tags)
    return encoded_labels, mlb


def split_data(lyrics, labels, test_size=0.2, random_state=42):
    """Split data into train and validation sets using stratified multilabel split."""
    lyrics_array = np.array(lyrics)
    labels_array = np.array(labels)
    
    # Use MultilabelStratifiedShuffleSplit for proper multi-label stratification
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    for train_idx, val_idx in msss.split(lyrics_array, labels_array):
        X_train = lyrics_array[train_idx].tolist()
        X_val = lyrics_array[val_idx].tolist()
        y_train = labels_array[train_idx]
        y_val = labels_array[val_idx]
        break
    
    return X_train, X_val, y_train, y_val


def stratified_multilabel_kfold(lyrics, labels, n_splits=5, random_state=42):
    """
    Performs a stratified k-fold split for multi-label data.
    
    Takes into account all labels of a sample, ensuring each label is 
    proportionally represented in each fold.
    
    Args:
        lyrics: List of lyrics texts
        labels: Multi-label encoded array (numpy array)
        n_splits: Number of folds
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Flatten multi-label structure for stratification
    flat_labels = []
    flat_indices = []

    for i, label_row in enumerate(labels):
        # Get indices of active labels
        active_labels = np.where(label_row == 1)[0]
        for label_idx in active_labels:
            flat_labels.append(label_idx)
            flat_indices.append(i)

    
    # Perform stratified split on flattened data
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True).split(X=[0]*len(flat_labels), y=flat_labels) # pyright: ignore[reportArgumentType]
    splits_ok = []
    for s in splits:
        ss = []
        for fold in s:
            ss.append(list(set([flat_indices[i] for i in fold])))
        splits_ok.append(ss)

    return splits_ok


def save_label_mapping(mlb, output_path):
    """Save label mapping to JSON file."""
    label_mapping = {i: label for i, label in enumerate(mlb.classes_)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    return label_mapping


def load_label_mapping(path):
    """Load label mapping from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def stratified_subset(lyrics, labels, subset_size=0.1, random_state=42):
    """
    Create a stratified subset of multi-label data for fast training.
    
    Args:
        lyrics: List of lyrics texts
        labels: Multi-label encoded array
        subset_size: Fraction of data to keep (default 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        Subset of lyrics and labels
    """
    if subset_size >= 1.0:
        return lyrics, labels
    
    # Convert to numpy arrays if needed
    lyrics_array = np.array(lyrics)
    labels_array = np.array(labels)
    
    # Use MultilabelStratifiedShuffleSplit for proper multi-label stratification
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=1-subset_size, 
        random_state=random_state
    )
    
    for subset_idx, _ in msss.split(lyrics_array, labels_array):
        subset_lyrics = lyrics_array[subset_idx].tolist()
        subset_labels = labels_array[subset_idx]
        break
    
    return subset_lyrics, subset_labels


def get_genre_pairs(tags):
    """
    Get all unique genre pairs from the dataset, excluding EXCLUDED_GENRES.
    
    Args:
        tags: List of tag lists for each song
    
    Returns:
        List of (genre1, genre2) tuples
    """
    all_genres = set()
    for tag_list in tags:
        for tag in tag_list:
            if tag not in EXCLUDED_GENRES:
                all_genres.add(tag)
    return list(combinations(sorted(all_genres), 2))


def filter_data_for_pair(lyrics, tags, genre_a, genre_b):
    """
    Filter songs that have exactly one of genre_a or genre_b (not both).
    
    Args:
        lyrics: List of lyrics texts (title + lyrics)
        tags: List of tag lists for each song
        genre_a: First genre
        genre_b: Second genre
    
    Returns:
        filtered_lyrics: List of lyrics for songs matching the criteria
        binary_labels: numpy array of 0 (genre_a) or 1 (genre_b)
    """
    filtered_lyrics = []
    binary_labels = []
    
    for text, song_tags in zip(lyrics, tags):
        has_a = genre_a in song_tags
        has_b = genre_b in song_tags
        
        # Include only if exactly one of the two genres is present
        if has_a and not has_b:
            filtered_lyrics.append(text)
            binary_labels.append(0)
        elif has_b and not has_a:
            filtered_lyrics.append(text)
            binary_labels.append(1)
    
    return filtered_lyrics, np.array(binary_labels)


class LyricsDataset(Dataset):
    """Dataset for lyrics multi-label classification."""
    
    def __init__(self, lyrics, labels, tokenizer, max_length=512):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        text = self.lyrics[idx]
        text = text.replace('refrén', ' ').replace('refr', ' ').replace('ref', ' ')
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


class BinaryLyricsDataset(Dataset):
    """Dataset for binary (one-vs-one) classification."""
    
    def __init__(self, lyrics, labels, tokenizer, max_length=512):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.lyrics)
    
    def __getitem__(self, idx):
        text = self.lyrics[idx]
        text = text.replace('refrén', ' ').replace('refr', ' ').replace('ref', ' ')
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def log_class_distribution(y_train, y_val, mlb, fold_num=None):
    """
    Log the distribution of classes in train and validation sets.
    
    Args:
        y_train: Training labels (binary encoded)
        y_val: Validation labels (binary encoded)
        mlb: MultiLabelBinarizer with class names
        fold_num: Optional fold number for k-fold CV
    """
    fold_str = f"Fold {fold_num}" if fold_num is not None else "Split"
    print("\n" + "=" * 60)
    print(f"{fold_str} - Class Distribution")
    print("=" * 60)
    print(f"{'Class':<20} {'Train':>10} {'Val':>10} {'Train %':>10} {'Val %':>10}")
    print("-" * 60)
    
    train_counts = y_train.sum(axis=0)
    val_counts = y_val.sum(axis=0)
    train_total = len(y_train)
    val_total = len(y_val)
    
    for i, class_name in enumerate(mlb.classes_):
        train_pct = (train_counts[i] / train_total) * 100
        val_pct = (val_counts[i] / val_total) * 100
        print(f"{class_name:<20} {int(train_counts[i]):>10} {int(val_counts[i]):>10} {train_pct:>9.1f}% {val_pct:>9.1f}%")
    
    print("-" * 60)
    print(f"{'Total samples':<20} {train_total:>10} {val_total:>10}")
    print("=" * 60 + "\n")