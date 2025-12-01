from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    root = Path(__file__).resolve()
    while root.name != "gen-cov-abm":  # replace with your repo root name
        root = root.parent
    return root


def get_data_dir(subfolder=None) -> Path:
    """Return absolute path to the data directory."""
    root = get_project_root()
    data_path = root / "data"
    return data_path / subfolder if subfolder else data_path
