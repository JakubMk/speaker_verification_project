import pytest
from pathlib import Path
from src.data.download_vc_dataset import download_zip, get_file_size, extract_id
from src.data import wav_to_spectrogram

def test_download_zip_file_not_exist(tmp_path):
    download_links = tmp_path / "non_existent_file.csv"
    dataset_dir = tmp_path / "dataset"

    with pytest.raises(FileNotFoundError):
        download_zip(download_links, dataset_dir)

def test_download_zip_file_bad_extension(tmp_path, bad_link):
    download_links = tmp_path / "existing_file_bad_extension.json"
    download_links.write_text(bad_link, encoding="utf-8")
    dataset_dir = tmp_path / "dataset"

    with pytest.raises(ValueError):
        download_zip(download_links, dataset_dir)

def test_download_zip_bad_link(tmp_path, bad_link, capsys):
    download_links = tmp_path / "existing_file.csv"
    download_links.write_text(bad_link, encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    download_zip(download_links, dataset_dir)
    out = capsys.readouterr().out
    assert "404 -- Not Found" in out

def test_get_file_size(tmp_path):
    download_links = tmp_path / "non_existent_file.csv"

    with pytest.raises(FileNotFoundError):
        get_file_size(download_links)


def test_extract_id(tmp_path):

    audio_file = Path(tmp_path.parts[-1]) / "audio1.wav"

    with pytest.raises(ValueError):
        extract_id(audio_file)

@pytest.mark.parametrize("preprocess_audio", [True, False])
def test_wav_to_spectrogram(mock_wav, preprocess_audio):
    spectrogram = wav_to_spectrogram(str(mock_wav), preprocess_audio=preprocess_audio)
    assert spectrogram.shape[0] == 512
    assert spectrogram.shape[1] == 300
    assert spectrogram.shape[2] == 3
    assert spectrogram.dtype == 'float32'