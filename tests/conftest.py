import pytest
import numpy as np
import soundfile as sf

@pytest.fixture
def bad_link():
    return """https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_dev_aac_partafgh"""

@pytest.fixture
def mock_csv(tmp_path):
    file = tmp_path / "mock_dataset.csv"
    file.write_text(
        """id,utt,file_path,duration_sec
        id00012,21Uxsk56VDQ,dataset/wav/id00012/21Uxsk56VDQ/00001.wav,9.408
        id00012,21Uxsk56VDQ,dataset/wav/id00012/21Uxsk56VDQ/00002.wav,15.296
        """,
        encoding="utf-8")
    return file

@pytest.fixture
def mock_wav(tmp_path):
    file = tmp_path / "mock_audio.wav"
    fs = 16000
    N = fs * 4 # 4 seconds
    t = np.linspace(0, 4, N, endpoint=False, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * 440 * t) # A4 note
    with sf.SoundFile(file, 'w', 16000, 1, 'PCM_16') as f:
        f.write(x)
    return file
