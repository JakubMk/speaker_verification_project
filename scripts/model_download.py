import argparse
from pathlib import Path
from src.utils import download_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained model from Google Drive.")
    parser.add_argument("--file_id", type=str, required=True, help="Google Drive file ID.")
    parser.add_argument("--output", type=str, default="models/model.keras", help="Where to save the model file.")
    args = parser.parse_args()
    download_model(args.file_id, Path(args.output))