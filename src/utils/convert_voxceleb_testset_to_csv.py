from pathlib import Path
import pandas as pd

def convert_voxceleb_testset_to_csv(
    input_txt: Path,
    dataset_dir: Path = Path("dataset"),
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Converts VoxCeleb official test .txt file to a model-ready CSV.
    """
    test_set = pd.read_csv(input_txt, sep=" ", header=None)
    test_set = test_set.rename(columns={0: 'y_true', 1: 'speaker_1', 2: 'speaker_2'})[['speaker_1', 'speaker_2', 'y_true']]
    test_set['speaker_1'] = test_set['speaker_1'].apply(lambda p: str(Path(dataset_dir) / 'wav' / p))
    test_set['speaker_2'] = test_set['speaker_2'].apply(lambda p: str(Path(dataset_dir) / 'wav' / p))
    if output_csv is not None:
        test_set.to_csv(output_csv, index=False)
    return test_set
