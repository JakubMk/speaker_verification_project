import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import random
import os
from src.data.train_dataset import delta, get_label, wav_to_spectrogram

def sample_diff_pos_utts(speaker_df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """
    Samples up to `n` unique utterances for a given speaker, ensuring each utterance (utt) is different if possible.

    Args:
        speaker_df (pd.DataFrame): DataFrame containing at least columns ['file_path', 'utt'] for a single speaker.
        n (int): Number of positive samples to draw.

    Returns:
        pd.DataFrame: Sampled DataFrame of size `n` (with replacement if insufficient unique utterances).
    """
    if len(speaker_df) <= n:
        return speaker_df.sample(n, replace=True, ignore_index=True)

    # create dict of {unique utt: [utterance files]}
    utt_dict = defaultdict(list)
    for _, row in speaker_df.iterrows():
        utt_dict[row['utt']].append(row['file_path'])

    # random shuffle utternaces
    utt_pool = list(utt_dict.keys())
    random.shuffle(utt_pool)

    results = []
    used_paths = set()

    while len(results) < n:
        for utt in utt_pool:
            available = [path for path in utt_dict[utt] if path not in used_paths]
            if not available:
                continue
            path = random.choice(available)
            used_paths.add(path)
            results.append(path)
            if len(results) == n:
                break

    return speaker_df[speaker_df['file_path'].isin(results)].reset_index(drop=True)

def generate_contrastive_dataset(
    df: pd.DataFrame, 
    n_pos_samples: int = 4, 
    n_neg_samples: int = 6
) -> pd.DataFrame:
    """
    Generates a single batch-wise DataFrame for contrastive training: 
    for each speaker, samples positive and negative examples.

    Args:
        df (pd.DataFrame): Must contain ['id', 'utt', 'file_path'] columns.
        n_pos_samples (int): Number of positive samples (from same speaker).
        n_neg_samples (int): Number of negative samples (from different speakers).

    Returns:
        pd.DataFrame: A DataFrame for a single batch (mixed positive and negative samples).
    """
    speaker_groups = {k: v for k, v in df.groupby('id')}
    all_speakers = list(speaker_groups.keys())
    output = []

    for speaker in tqdm(all_speakers, desc="Sampling contrastive batch"):
        # sample NO_OF_POS_SPEAKERS of different utts
        pos_samples = sample_diff_pos_utts(speaker_groups[speaker], n=n_pos_samples)

        # sample NO_OF_NEG_SPEAKERS
        neg_speakers = [s for s in all_speakers if s != speaker]
        neg_speakers_sampled = random.sample(neg_speakers, n_neg_samples)
        neg_samples = pd.concat([speaker_groups[s].sample(1) for s in neg_speakers_sampled])

        # concatenate negative and positive
        batch_df = pd.concat([pos_samples, neg_samples], ignore_index=True)
        output.append(batch_df)

    final_df = pd.concat(output, ignore_index=True)
    return final_df

def contrastive_dataset(
    dataset_df: pd.DataFrame,
    num_epochs: int = 1,
    n_pos_samples: int = 4,
    n_neg_samples: int = 6,
    batch_size: int = 8
) -> tuple[tf.data.Dataset, np.ndarray]:
    """
    Generates a contrastive training `tf.data.Dataset` by repeatedly sampling positive/negative batches per speaker.
    
    For each epoch, this function generates a new set of positive and negative pairs for every speaker, 
    resulting in a new augmented dataset per epoch. All epochs are concatenated into one large dataset 
    (not a tf.data.Dataset with multiple epochs).

    This design is intentional: **model.fit() should always use epochs=1** when training on this dataset, 
    since we pre-generate all batches for the desired number of epochs in advance. 
    This is required because dynamic contrastive sampling is not possible with model.fit() 
    (Keras expects finite, repeatable datasets).

    Args:
        dataset_df (pd.DataFrame): DataFrame with ['id', 'utt', 'file_path'] columns.
        num_epochs (int): How many times to generate batches (epochs).
        n_pos_samples (int): Positive samples per speaker per batch.
        n_neg_samples (int): Negative samples per speaker per batch.
        batch_size (int): Batch size for tf.data.Dataset.

    Returns:
        tuple: (tf.data.Dataset, np.ndarray)
            - dataset: tf.data.Dataset yielding (spectrogram, label) pairs
            - classes: np.ndarray of unique class labels
    """
    dataset_parts = [
        generate_contrastive_dataset(dataset_df, n_pos_samples, n_neg_samples)
        for _ in range(num_epochs)
    ]
    dataset_df = pd.concat(dataset_parts, ignore_index=True)
    classes = np.unique(dataset_df.id)
    classes_tf = tf.constant(classes)

    def process_path(file_path):
        label = get_label(file_path, classes_tf)
        spectrogram = wav_to_spectrogram(file_path)
        return spectrogram, label

    AUTOTUNE = tf.data.AUTOTUNE
    dataset_file_paths = dataset_df.file_path.astype(str).values
    contrastive_ds = tf.data.Dataset.from_tensor_slices(dataset_file_paths)
    contrastive_ds = contrastive_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    contrastive_ds = contrastive_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return contrastive_ds, classes
