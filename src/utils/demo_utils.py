import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import ffmpeg
import numpy as np
from pathlib import Path
import tempfile
import tensorflow as tf
from src.data import delta
from src.models import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

def wav_to_full_spectrogram(
    file_path,
    window_length=400,
    step_length=160,
    fft_length=1023
):
    """
    Loads an audio file, computes log-magnitude spectrogram and its first and second derivatives
    (delta, delta^2) using the FULL audio length.
    Output shape: [512, N, 3], where N is the number of time frames.
    """
    # Read the file
    example = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(
        example,
        desired_channels=1,
        desired_samples=-1,
    )

    audio = tf.squeeze(audio, axis=-1)

    stft = tf.signal.stft(audio,
                          frame_length=window_length,
                          frame_step=step_length,
                          fft_length=fft_length)
    
    spectrogram = tf.abs(stft)
    spectrogram = tf.transpose(spectrogram)  # shape: (freq, time)
    spectrogram = tf.math.log1p(spectrogram)

    spectrogram_delta = delta(spectrogram)
    spectrogram_delta2 = delta(spectrogram_delta)

    min_time = min(
        spectrogram.shape[1],
        spectrogram_delta.shape[1],
        spectrogram_delta2.shape[1]
    )

    return tf.stack([
        spectrogram[:, :min_time],
        spectrogram_delta[:, :min_time],
        spectrogram_delta2[:, :min_time]
    ], axis=-1)  # shape: (512, min_time, 3)

def convert_to_wav(input_path: str | Path) -> tuple[Path, bool]:
    """
    Converts a non-WAV audio file to WAV format using ffmpeg and returns a Path to the new file.

    Parameters
    ----------
    input_path : Path or str
        Path to the input audio file. Supported formats: .wav, .m4a, .aac, .mp3, .ogg

    Returns
    -------
    output_path : Path
        Path to a WAV file (original if already WAV, or converted temporary file if not).
    is_temp : bool
        True if a temporary WAV file was created and should be deleted after use.
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext in ['.m4a', '.aac', '.mp3', '.ogg']:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            ffmpeg.input(str(input_path)).output(tmpfile.name, ar=16000, ac=1).run(quiet=True, overwrite_output=True)
            return Path(tmpfile.name), True
    else:
        return input_path, False

def verify_speaker(
    model_path: str | Path,
    recording_1: str | Path,
    recording_2: str | Path,
    margin: float = 0.3050
) -> float:
    """
    Compares two audio recordings using a speaker verification model.
    Automatically converts non-WAV files to WAV using ffmpeg, extracts spectrograms,
    computes cosine similarity, prints a decision, and cleans up temporary files.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved Keras model (.keras file).
    recording_1 : str or Path
        Path to the first audio file (WAV, M4A, AAC, MP3, OGG).
    recording_2 : str or Path
        Path to the second audio file (WAV, M4A, AAC, MP3, OGG).
    margin : float, optional
        Cosine similarity threshold for positive decision (default: 0.25).

    Returns
    -------
    cosine_similarity : float
        Cosine similarity between embeddings.
    """
    for path, label in zip([model_path, recording_1, recording_2], ["model_path", "recording_1", "recording_2"]):
        if not Path(path).is_file():
            raise FileNotFoundError(f"File not found: {label} -> '{path}'")
        
    # Load model
    model = tf.keras.models.load_model(model_path)

    # If model has 'return_embedding' attribute
    if hasattr(model, 'return_embedding'):
        model.return_embedding = True

    recording_1, temp1 = convert_to_wav(recording_1)
    recording_2, temp2 = convert_to_wav(recording_2)

    spectrogram_1 = wav_to_full_spectrogram(str(recording_1))
    spectrogram_2 = wav_to_full_spectrogram(str(recording_2))

    emb_1 = model(np.expand_dims(spectrogram_1, axis=0))
    emb_2 = model(np.expand_dims(spectrogram_2, axis=0))

    cosine_similarity = tf.linalg.matmul(emb_1, emb_2, transpose_b=True)
    cosine_similarity = float(cosine_similarity.numpy().squeeze())

    if cosine_similarity >= margin:
        result = "It's the same person!"
    else:
        result = "It's not the same person."
    print(f"Cosine similarity: {cosine_similarity:.4f} -> {result}")

    # Clean up temporary files if needed
    if temp1 and recording_1.exists():
        recording_1.unlink()
    if temp2 and recording_2.exists():
        recording_2.unlink()
        
    return cosine_similarity
