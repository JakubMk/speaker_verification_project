import argparse
from pathlib import Path
from src.utils import convert_voxceleb_testset_to_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VoxCeleb official testset .txt to model-ready CSV."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to original test .txt file (e.g. list_test_all2.txt)"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset",
        help="Root directory where audio files are stored (prefix for file paths)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    input_txt = Path(args.input)
    dataset_dir = Path(args.dataset_dir)
    output_csv = Path(args.output)

    print(f"Converting {input_txt} to {output_csv} (dataset root: {dataset_dir})")
    df = convert_voxceleb_testset_to_csv(input_txt, dataset_dir, output_csv)
    print(f"Done! Output: {output_csv} | Shape: {df.shape}")
