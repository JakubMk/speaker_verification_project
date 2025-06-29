import gdown
from pathlib import Path

def download_pretrained_model(file_id: str, out_path: Path = Path("models/model.keras")):
    """
    Downloads a file from Google Drive by its file_id and saves it to out_path.

    Args:
        file_id (str): Google Drive file ID (see Google Drive share link).
        out_path (Path, optional): Output path for the downloaded file (default: models/model.keras).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out_path), quiet=False)
