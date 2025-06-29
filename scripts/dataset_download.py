import argparse
from src.data import download_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and prepare VoxCeleb1 or VoxCeleb2 dataset.")
    parser.add_argument(
        "--dataset", type=int, choices=[1, 2], required=True,
        help="Select dataset to download: 1 for VoxCeleb1, 2 for VoxCeleb2."
    )
    parser.add_argument(
        "--csv_dir", type=str, required=True,
        help="Directory containing download link CSVs and where output filelists will be saved."
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="dataset",
        help="Directory where audio files will be stored. Default: 'dataset'"
    )
    parser.add_argument(
        "--remove_parts", action="store_true", default=False,
        help="Remove ZIP part files after extraction."
    )
    parser.add_argument(
        "--remove_extracted", action="store_true", default=False,
        help="Remove ZIP files after extraction."
    )

    args = parser.parse_args()

    dev_df, test_df = download_dataset(
        vox_celeb_dataset=args.dataset,
        csv_dir=args.csv_dir,
        dataset_dir=args.dataset_dir,
        remove_parts=args.remove_parts,
        remove_extracted=args.remove_extracted
    )
