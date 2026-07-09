"""Path resolution utility for the Vardan research project.

Exposes Path objects relative to the project root, ensuring compatibility across
Windows, Linux, and macOS without modifying source code.
"""

from pathlib import Path

# PROJECT_ROOT resolves to the main 'Vardan/' workspace directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Subdirectories
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPERS_DIR = PROJECT_ROOT / "papers"


def ensure_directories():
    """Verify and create directory structure (excluding git-ignored content if raw)."""
    directories = [
        CONFIGS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOK_DIR,
        MODELS_DIR,
        MODELS_DIR / "baselines",
        MODELS_DIR / "vardhan",
        MODELS_DIR / "checkpoints",
        EXPERIMENTS_DIR,
        EXPERIMENTS_DIR / "configs",
        EXPERIMENTS_DIR / "logs",
        EXPERIMENTS_DIR / "runs",
        EXPERIMENTS_DIR / "reports",
        FIGURES_DIR,
        RESULTS_DIR,
        RESULTS_DIR / "tables",
        RESULTS_DIR / "plots",
        RESULTS_DIR / "metrics",
        RESULTS_DIR / "confusion_matrices",
        REPORTS_DIR,
        REPORTS_DIR / "literature",
        REPORTS_DIR / "notes",
        REPORTS_DIR / "weekly_logs",
        REPORTS_DIR / "paper",
        PAPERS_DIR,
        PAPERS_DIR / "Dataset_Papers",
        PAPERS_DIR / "RF_Papers",
        PAPERS_DIR / "Acoustic_Papers",
        PAPERS_DIR / "Vision_Papers",
        PAPERS_DIR / "Surveys",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Project directory structure verified/created.")
    print(f"Project root is: {PROJECT_ROOT}")
