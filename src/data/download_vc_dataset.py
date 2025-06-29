import ffmpeg
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import pandas as pd
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import zipfile
import shutil
import sys
tqdm.pandas(desc="Checking audio")


def check_ffmpeg():
    """
    Checks if ffmpeg is installed and available in PATH.
    Exits the program if ffmpeg is not found.
    """
    if shutil.which("ffmpeg") is None:
        print(
            "\nERROR: ffmpeg is NOT installed or not available in your system PATH.\n"
            "Please install ffmpeg and make sure it is accessible from the command line.\n"
            "See installation instructions in the README (https://ffmpeg.org/download.html).\n"
        )
        sys.exit(1)
    else:
        print("ffmpeg found in PATH.\n")


def download_zip(download_links: Path, dataset_dir: Path) -> Path:
    """
    Downloads files from URLs listed in a text file to a specified directory.

    Args:
        download_links (Path): Path to a text file containing one URL per line.
        dataset_dir (Path): Destination directory to save downloaded files.

    Returns:
        Path: Path to the directory where files are downloaded.

    Raises:
        FileNotFoundError: If the download_links file does not exist.
        Exception: If a download fails for other reasons.
    """
    links = []
    dataset_dir = Path(dataset_dir)

    if not (download_links.exists() and download_links.is_file()):
        raise FileNotFoundError(f"Download links file '{download_links}' does not exist.")
    
    if not dataset_dir.exists():
        print(f"Dataset directory '{dataset_dir}' does not exist. Creating it now.\n")
        dataset_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Dataset directory '{dataset_dir}' already exists.\n")

    print(f"Reading download links from '{download_links}'.\n")
    with open(download_links, 'r', encoding='utf-8') as file:
        for line in file:
            url = line.strip()
            if url:
                links.append(url)

    if any(dataset_dir.iterdir()):
        print(f"Dataset directory '{dataset_dir}' already exists and is not empty.\n")
        user_input = input(f"Do you want to download files to '{dataset_dir}' anyway? [y/N]: ").strip().lower()
        if user_input not in ['y', 'yes']:
            print("Download aborted by user.")
            return dataset_dir
        else:
            print(f"Downloading files to '{dataset_dir}' anyway.\n")
    else:
        print(f"Dataset directory '{dataset_dir}' is empty. Downloading files...\n")
    
    for link in links:
        try:
            tf.keras.utils.get_file(
                origin=link,
                extract=False,
                cache_dir=str(dataset_dir.parent),
                cache_subdir=dataset_dir.name)
            print(f"Downloaded: {link}")
        except Exception as e:
            print(f"Failed to download {link}. Error: {e}")
    
    return dataset_dir


def get_file_size(file_path: Path) -> int:
    """
    Returns the size of the given file in bytes.

    Args:
        file_path (Path): Path to the file.

    Returns:
        int: Size of the file in bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return file_path.stat().st_size


def copy_file_with_pbar(src: Path, dest_file, chunk_size: int) -> int:
    """
    Copies a file in chunks with a progress bar.

    Args:
        src (Path): Path to the source file.
        dest_file (file object): Open file object to write the data to (in binary mode).
        chunk_size (int): Number of bytes to read/write at a time.

    Returns:
        int: Total number of bytes written.

    Raises:
        FileNotFoundError: If the source file does not exist.
        Exception: For general I/O errors.
    """
    file_size = get_file_size(src)
    bytes_written = 0

    with open(src, 'rb') as input_file:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=src.name) as pbar:
            while chunk := input_file.read(chunk_size):
                dest_file.write(chunk)
                bytes_written += len(chunk)
                pbar.update(len(chunk))
    return bytes_written


def concatenate_file_parts(output_zip: Path,
                      parts_list: list[Path],
                      chunk_size: int = 64 * 1024 * 1024,
                      remove_parts: bool = False) -> Path:
    """
    Concatenates multiple file parts into a single output file.

    Args:
        output_zip (Path): Path to the final output file.
        parts_list (list[Path]): List of Path objects pointing to file parts (in order).
        chunk_size (int, optional): Number of bytes to read/write at a time. Default is 64 MB.
        remove_parts (bool, optional): Whether to remove parts after concatenation. Default is False.

    Returns:
        Path: Path to the concatenated output file.

    Raises:
        FileNotFoundError: If parts_list is empty.
        Exception: For general I/O errors.
    """
    if not parts_list:
        raise FileNotFoundError("No parts found to concatenate")

    try:
        with open(output_zip, 'wb') as output_file:
            for part_file in parts_list:
                try:
                    print(f"Concatenating {part_file.name}...")
                    bytes_written = copy_file_with_pbar(part_file, output_file, chunk_size)
                    print(f"Done with {part_file.name} ({bytes_written / (1024 ** 2):.2f} MB)")
                    if remove_parts:
                        part_file.unlink()
                except Exception as e:
                    print(f"Error while processing {part_file.name}: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    return output_zip


def extract_zip(
    zip_filepath: Path,
    extracted_folder_name: Path,
    remove_extracted: bool = True
) -> pd.DataFrame:
    """
    Extracts audio files (.wav, .m4a) from a zip archive and returns a DataFrame with their paths.

    Args:
        zip_filepath (Path): Path to the zip file to extract.
        extracted_folder_name (Path): Path to the destination directory where files will be extracted.
        remove_extracted (bool, optional): If True, deletes the zip file after extraction. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with a single column 'file_path' containing paths to extracted audio files.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        Exception: For general extraction errors.
    """
    extracted_files = []
    allowed_extensions = {'.wav', '.m4a'}
    
    if not zip_filepath.exists():
        raise FileNotFoundError(f"Zip file {zip_filepath} does not exist.")

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc="Extracting files"):
            try:
                zip_ref.extract(member, extracted_folder_name)
                extracted_path = Path(extracted_folder_name) / member
                # append only .wav or .aac files
                if extracted_path.suffix.lower() in allowed_extensions:
                    extracted_files.append(extracted_path.resolve())
            except Exception as e:
                print(f"Error during extracting {member}: {e}")
    print(f'Files extracted to: {extracted_folder_name}')
    if remove_extracted:
        try:
            zip_filepath.unlink()
        except Exception as e:
            print(f"Error removing zip file: {e}")

    return pd.DataFrame({'file_path': extracted_files})


def convert_path_m4a_to_wav(path: Path) -> Path:
    """
    Converts a Path from a 'dev/aac' structure to a 'wav' structure and changes the file extension to .wav.

    Removes the 'dev' segment from the path (if present), replaces the first occurrence of 'aac' with 'wav',
    and ensures the file extension is '.wav'.

    Args:
        path (Path): Original file path (potentially containing 'dev' and 'aac' in its structure).

    Returns:
        Path: Converted file path with 'wav' in place of 'aac', 'dev' removed, and a '.wav' extension.
    """
    new_parts = []
    for part in path.parts:
        if part == 'dev':
            continue  # skip 'dev'
        elif part == 'aac':
            part = 'wav'
        else:
            part = part
        
        new_parts.append(part)
    return Path(*new_parts).with_suffix(".wav")


def convert_audio_ffmpeg(row: dict) -> int:
    """
    Converts a single audio file to WAV format using ffmpeg.

    The input file is read from 'file_path' and saved as 'wav_file_path' (both in the row dict).
    The function removes the original input file after successful conversion.

    Args:
        row (dict): Dictionary with keys 'file_path' and 'wav_file_path'.

    Returns:
        int: 1 if conversion succeeded, 0 otherwise.
    """
    try:
        input_file = Path(row['file_path'])
        output_file = Path(row['wav_file_path'])
        output_file.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg.input(str(input_file)).output(str(output_file)).run(quiet=True, overwrite_output=True)
        if input_file.exists():
            input_file.unlink()
        return 1
    except Exception as e:
        print(f"File error occurred {row['file_path']}: {e}")
        return 0


def convert_dataset(df: pd.DataFrame, max_workers: int = None) -> pd.Series:
    """
    Converts a dataset of audio files to WAV format in parallel using ffmpeg.

    Each row of the dataframe must contain 'file_path' and 'wav_file_path' columns.

    Args:
        df (pd.DataFrame): DataFrame with columns 'file_path' and 'wav_file_path'.
        max_workers (int, optional): The maximum number of threads to use. Defaults to None (auto).

    Returns:
        pd.Series: Series of integers (1 if conversion succeeded, 0 otherwise) in the same order as df.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(convert_audio_ffmpeg, df.to_dict('records')),
            total=len(df),
            desc="Converting files"
        ))
    return pd.Series(results)


def check_audio_and_length(audio_path: Path) -> pd.Series:
    """
    Checks if the given audio file can be read and returns its duration in seconds.

    Args:
        audio_path (Path): Path to the audio file.

    Returns:
        pd.Series: Series with fields:
            - 'read_ok' (int): 1 if file can be read, 0 otherwise.
            - 'duration_sec' (float or None): Duration of the audio file in seconds, or None if unreadable.
    """
    try:
        with sf.SoundFile(audio_path) as f:
            length_sec = len(f) / f.samplerate
            return pd.Series({'read_ok': 1, 'duration_sec': length_sec})
    except Exception:
        return pd.Series({'read_ok': 0, 'duration_sec': None})


def extract_id(audio_path: Path) -> str:
    """
    Extracts the speaker ID (class) from the audio file path.
    
    Assumes that the speaker ID is located at the third position from the end of the path.

    Args:
        audio_path (Path): The full path to the audio file.

    Returns:
        str: The extracted speaker ID.
    """
    return audio_path.parts[-3]


def extract_utt(audio_path: Path) -> str:
    """
    Extracts the utterance ID from the audio file path.
    
    Assumes that the utterance ID is located at the second position from the end of the path.

    Args:
        audio_path (Path): The full path to the audio file.

    Returns:
        str: The extracted utterance ID.
    """
    return audio_path.parts[-2]


def download_dataset(vox_celeb_dataset: int,
                     csv_dir: str,
                     dataset_dir: str = 'dataset',
                     remove_parts: bool = False,
                     remove_extracted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads and prepares the selected VoxCeleb dataset (VoxCeleb1 or VoxCeleb2).

    Depending on the chosen dataset, this function downloads audio files (via links in a CSV),
    extracts and processes the data, performs audio integrity checks, converts files to WAV (for VoxCeleb2),
    removes corrupted files, and generates clean filelists as CSV for downstream training.

    Args:
        vox_celeb_dataset (int): 1 for VoxCeleb1, 2 for VoxCeleb2.
        csv_dir (str): Directory containing the dataset download link CSV files and
            where the resulting CSV filelists will be saved.
        dataset_dir (str, optional): Directory where the dataset will be extracted. Defaults to 'dataset'.
        remove_parts (bool, optional): If True, remove ZIP part files after concatenation. Defaults to True.
        remove_extracted (bool, optional): If True, remove ZIP files after extraction. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of two DataFrames (dev_df, test_df) with clean filelists and metadata.

    Raises:
        ValueError: If an invalid dataset choice is provided.
        FileNotFoundError: If required files do not exist.
        Exception: If any step in the process fails.
    """
    if vox_celeb_dataset not in [1, 2]:
        raise ValueError("Invalid dataset choice. Use 1 for VoxCeleb1 or 2 for VoxCeleb2.")
    
    dataset_dir = Path(dataset_dir)
    
    if vox_celeb_dataset == 1:
        download_links = Path(csv_dir) / 'download_dataset_vc1.csv'
        zip_dev_name = 'vox1_dev_wav.zip'
        parts_dev_name = 'vox1_dev_wav_part'
        zip_test_name = 'vox1_test_wav.zip'
    elif vox_celeb_dataset == 2:
        check_ffmpeg()
        download_links = Path(csv_dir) / 'download_dataset_vc2.csv'
        zip_dev_name = 'vox2_dev_aac.zip'
        parts_dev_name = 'vox2_dev_aac_part'
        zip_test_name = 'vox2_test_aac.zip'

    dataset_dir = download_zip(download_links, dataset_dir)
    parts = sorted(dataset_dir.glob(parts_dev_name + '*'))
    output_zip = concatenate_file_parts(dataset_dir / zip_dev_name, parts, remove_parts=remove_parts)

    dev_df = extract_zip(output_zip, dataset_dir, remove_extracted=remove_extracted)
    dev_df['file_path'] = dev_df['file_path'].apply(lambda p: Path(*Path(p).parts[(-4-len(dataset_dir.parts)):]))
    dev_df = dev_df.sort_values(by='file_path').reset_index(drop=True)

    test_df = extract_zip(dataset_dir / zip_test_name, dataset_dir, remove_extracted=remove_extracted)
    test_df['file_path'] = test_df['file_path'].apply(lambda p: Path(*Path(p).parts[(-4-len(dataset_dir.parts)):]))
    test_df = test_df.sort_values(by='file_path').reset_index(drop=True)

    # Convert m4a files to wav format when using VoxCeleb2
    if vox_celeb_dataset == 1:

        dev_df[['read_ok', 'duration_sec']] = dev_df['file_path'].progress_apply(check_audio_and_length)
        test_df[['read_ok', 'duration_sec']] = test_df['file_path'].progress_apply(check_audio_and_length)

        print("dev set corrupted files: ", len(dev_df[dev_df['read_ok']==0]))
        print("test set corrupted files: ", len(test_df[test_df['read_ok']==0]))

        # remove corrupted files
        dev_df = dev_df[dev_df['read_ok'] == 1]
        test_df = test_df[test_df['read_ok'] == 1]

        # save file white-list to csv, no header, relative paths
        devset_whitelist_file = Path(csv_dir) / "vc1_dev_ok.csv"
        testset_whitelist_file = Path(csv_dir) / "vc1_test_ok.csv"

        # adding label columns to datasets and adding them to DataFrames
        dev_df['id'] = dev_df['file_path'].apply(extract_id)
        test_df['id'] =test_df['file_path'].apply(extract_id)

        dev_df['utt'] = dev_df['file_path'].apply(extract_utt)
        test_df['utt'] =test_df['file_path'].apply(extract_utt)

        # dropping read_ok column
        dev_df = dev_df.drop('read_ok', axis=1)
        test_df = test_df.drop('read_ok', axis=1)

        # sort columns
        dev_df = dev_df.reindex(columns=['id', 'utt', 'file_path', 'duration_sec'])
        test_df = test_df.reindex(columns=['id', 'utt', 'file_path', 'duration_sec'])

        dev_df.to_csv(devset_whitelist_file, header=True, index=False)
        test_df.to_csv(testset_whitelist_file, header=True, index=False)

    elif vox_celeb_dataset == 2:

        dev_df["wav_file_path"] = dev_df["file_path"].apply(convert_path_m4a_to_wav)
        test_df["wav_file_path"] = test_df["file_path"].apply(convert_path_m4a_to_wav)
        dev_df['to_wav_converted'] = convert_dataset(dev_df)
        test_df['to_wav_converted'] = convert_dataset(test_df)

        dev_df[['read_ok', 'duration_sec']] = dev_df['wav_file_path'].progress_apply(check_audio_and_length)
        test_df[['read_ok', 'duration_sec']] = test_df['wav_file_path'].progress_apply(check_audio_and_length)

        print("dev set corrupted files: ", len(dev_df[dev_df['read_ok']==0]))
        print("test set corrupted files: ", len(test_df[test_df['read_ok']==0]))

        # remove corrupted files
        dev_df = dev_df[dev_df['read_ok'] == 1]
        test_df = test_df[test_df['read_ok'] == 1]

        devset_whitelist_file = Path(csv_dir) / "vc2_dev_ok.csv"
        testset_whitelist_file = Path(csv_dir) / "vc2_test_ok.csv"

        # adding label columns to datasets and adding them to DataFrames
        dev_df['id'] = dev_df['wav_file_path'].apply(extract_id)
        test_df['id'] =test_df['wav_file_path'].apply(extract_id)

        dev_df['utt'] = dev_df['wav_file_path'].apply(extract_utt)
        test_df['utt'] =test_df['wav_file_path'].apply(extract_utt)

        # dropping columns
        dev_df = dev_df.drop(['read_ok', 'to_wav_converted', 'file_path'], axis=1)
        test_df = test_df.drop(['read_ok', 'to_wav_converted', 'file_path'], axis=1)

        # column rename
        dev_df = dev_df.rename(columns={'wav_file_path': 'file_path'})
        test_df = test_df.rename(columns={'wav_file_path': 'file_path'})

        # sort columns
        dev_df = dev_df.reindex(columns=['id', 'utt', 'file_path', 'duration_sec'])
        test_df = test_df.reindex(columns=['id', 'utt', 'file_path', 'duration_sec'])

        dev_df.to_csv(devset_whitelist_file, header=True, index=False)
        test_df.to_csv(testset_whitelist_file, header=True, index=False)
    
    print(f"Filelists saved to '{devset_whitelist_file}' and '{testset_whitelist_file}'.\n")
    
    return dev_df, test_df
