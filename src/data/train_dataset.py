import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Tuple, Optional

def delta(x):
    """Computes first-order difference along time axis."""
    return x[:, 1:] - x[:, :-1]

def get_label(file_path, classes):
    """
    Converts a file path into a class index based on folder name.
    
    Args:
        file_path: Path or tf.string tensor of the file.
        classes: Numpy array or tf.constant of all class labels.
    
    Returns:
        tf.Tensor: Index of the class as integer.
    """
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-3] == classes
    # Integer encode the label
    return tf.argmax(one_hot)

def pre_emphasis(signal: tf.Tensor, alpha: float = 0.97) -> tf.Tensor:
    """
    Apply pre‑emphasis (FIR filter) to an audio signal.

    Args:
    signal : tf.Tensor of shape (n_samples,), dtype float32/float64, obtained from `tf.audio.decode_wav`.
    alpha : float, optional. Default is 0.97.

    Returns:
    tf.Tensor: filtered signal with the same shape and dtype as the input.
    """
    # Keep the first sample unchanged
    first = signal[:1]

    # Apply pre‑emphasis to the rest of the signal
    rest = signal[1:] - alpha * signal[:-1]

    return tf.concat([first, rest], axis=0)

def wav_to_spectrogram(
    file_path,
    audio_in_samples=48560,
    window_length=400,
    step_length=160,
    fft_length=1023,
    preprocess_audio=False
):
    """
    Loads an audio file, extracts a random window, computes log-magnitude spectrogram
    and its first and second derivatives (delta, delta^2).
    Output shape: [512, 300, 3]
    """
    # Read the file
    example = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(
        example,
        desired_channels=1,
        desired_samples=-1,
    )
    audio = tf.squeeze(audio, axis=-1)
    audio_length = tf.shape(audio)[0]

    if preprocess_audio:
        audio = pre_emphasis(audio)
        peak = tf.reduce_max(tf.abs(audio))
        audio = tf.where(peak > 0, audio / peak, audio)

    # Sample random offset for a 3s window (if audio is long enough, in VC datasets it is)
    random_int = tf.random.uniform(shape=(), minval=0, maxval=(audio_length-audio_in_samples), dtype=tf.int32)
    stft = tf.signal.stft(audio[random_int:(random_int+audio_in_samples)],
                          frame_length=window_length,
                          frame_step=step_length,
                          fft_length=fft_length)
    
    spectrogram = tf.abs(stft)
    spectrogram = tf.transpose(spectrogram)  # shape: (freq, time)
    spectrogram = tf.math.log1p(spectrogram)

    spectrogram_delta = delta(spectrogram)
    spectrogram_delta2 = delta(spectrogram_delta)

    return tf.stack([spectrogram[:, :-2],
                     spectrogram_delta[:, :-1],
                     spectrogram_delta2],
                     axis=-1) # shape: (freq, time, 3)

def configure_for_performance(ds, batch_size=16, autotune=None):
    if autotune is None:
        autotune = tf.data.AUTOTUNE
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=autotune)
    return ds

def create_training_dataset(
    dev1_csv: str,
    dev2_csv: str,
    batch_size: int = 16,
    eer_val: bool = False,
    preprocess_audio: bool = False,
    autotune: int = tf.data.AUTOTUNE
) -> Tuple[np.ndarray, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Creates TensorFlow training/validation datasets from DataFrames.
    
    Args:
        dev1_df, dev2_df: filepaths to Pandas DataFrames with at least `id` and `file_path` columns.
        batch_size: Batch size for tf.data.Dataset
        eer_val: If False, creates a validation set with the longest sample per class.
        normalize_audio: If True, normalizes audio file before processing.
        autotune: Autotune setting for parallel processing in tf.data.Dataset

    Returns:
        classes: np.ndarray of class labels
        train_ds: tf.data.Dataset for training
        val_ds: tf.data.Dataset for validation (if eer_val is False)
    """
    dev1_df  = pd.read_csv(dev1_csv)
    dev2_df = pd.read_csv(dev2_csv)

    df = pd.concat([dev1_df, dev2_df], ignore_index=True)
    classes = np.unique(df.id)
    classes_tf = tf.constant(classes)

    def process_path(file_path):
        label = get_label(file_path, classes_tf)
        spectrogram = wav_to_spectrogram(file_path, preprocess_audio=preprocess_audio)
        return spectrogram, label

    if not eer_val:
        val_df = df.sort_values('duration_sec', ascending=False, ignore_index=True).drop_duplicates('id', ignore_index=True)
        train_df = df[~df.file_path.isin(val_df.file_path)]
        validation_file_paths = val_df.file_path.values
        validation_ds = tf.data.Dataset.from_tensor_slices(validation_file_paths)
        validation_ds = validation_ds.shuffle(len(validation_file_paths), reshuffle_each_iteration=True)
        validation_ds = validation_ds.map(process_path, num_parallel_calls=autotune)
        validation_ds = configure_for_performance(validation_ds, batch_size, autotune=autotune)

        train_file_paths = train_df.file_path.values
        train_ds = tf.data.Dataset.from_tensor_slices(train_file_paths)
        train_ds = train_ds.shuffle(len(train_file_paths), reshuffle_each_iteration=True)
        train_ds = train_ds.map(process_path, num_parallel_calls=autotune)
        train_ds = configure_for_performance(train_ds, batch_size, autotune=autotune)

        return classes, train_ds, validation_ds
    else:
        train_file_paths = df.file_path.values
        train_ds = tf.data.Dataset.from_tensor_slices(train_file_paths)
        train_ds = train_ds.shuffle(len(train_file_paths), reshuffle_each_iteration=True)
        train_ds = train_ds.map(process_path, num_parallel_calls=autotune)
        train_ds = configure_for_performance(train_ds, batch_size, autotune=autotune)
       
        return classes, train_ds, None

def test_dataset(
    test_df: pd.DataFrame,
    batch_size: int = 16,
    autotune: int = tf.data.AUTOTUNE
) -> tuple[tf.data.Dataset, np.ndarray]:
    """
    Creates a TensorFlow Dataset for evaluation from a test DataFrame containing file path pairs.

    The function extracts all unique audio file paths from the 'speaker_1' and 'speaker_2' columns,
    computes their spectrograms, and prepares the batched dataset for efficient inference.

    Args:
        test_df (pd.DataFrame): DataFrame with columns ['speaker_1', 'speaker_2'] listing test file paths.
        batch_size (int, optional): Batch size for the returned dataset. Defaults to 16.
        autotune (int, optional): Value for TensorFlow's autotune parallel calls. Defaults to tf.data.AUTOTUNE.

    Returns:
        tuple[tf.data.Dataset, np.ndarray]: 
            - tf.data.Dataset: Batched, prefetched dataset of spectrograms for evaluation.
            - np.ndarray: Array of all unique test file paths.
    """
    if autotune is None:
        autotune = tf.data.AUTOTUNE
    all_test_paths = np.unique(test_df[['speaker_1', 'speaker_2']].values.ravel())
    test_ds = tf.data.Dataset.from_tensor_slices(all_test_paths)
    test_ds = test_ds.map(wav_to_spectrogram, num_parallel_calls=autotune)
    test_ds = configure_for_performance(test_ds, batch_size, autotune=autotune)
    
    return test_ds, all_test_paths
