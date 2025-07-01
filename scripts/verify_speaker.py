import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from pathlib import Path
from src.utils import verify_speaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Verification CLI")
    parser.add_argument('--model', type=str, required=True, help='Path to Keras model (.keras)')
    parser.add_argument('--rec1', type=str, required=True, help='Path to first audio file')
    parser.add_argument('--rec2', type=str, required=True, help='Path to second audio file')
    parser.add_argument('--margin', type=float, default=0.3050, help='Cosine similarity threshold (default: 0.3050)')

    args = parser.parse_args()

    similarity = verify_speaker(
        model_path=args.model,
        recording_1=args.rec1,
        recording_2=args.rec2,
        margin=args.margin
    )
