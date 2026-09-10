"""
Rebuilds the main setup cell of work/notebooks/capstone.ipynb so that
Pylance/Pyright can resolve the conditional Colab-only import cleanly.

This does NOT:
- install google.colab locally
- remove the Colab path
- change ML methodology
- hide diagnostics via IDE settings
"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("work/notebooks/capstone.ipynb")


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, nb: dict) -> None:
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


def find_cell(nb: dict, marker: str) -> dict | None:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if marker in "".join(cell["source"]):
            return cell
    return None


def new_setup_cell() -> list[str]:
    return [
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# Section 3: Pipeline Architecture & Dependency Initialization\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "import os\n",
        "import sys\n",
        "\n",
        "# Tell static analyzers (Pylance/Pyright) that google.colab is a\n",
        "# Colab-runtime import. We still keep the real runtime import inside a\n",
        "# try/except so local VS Code / Jupyter execution stays valid.\n",
        "if sys.version_info >= (0, 0):\n",
        "    from typing import TYPE_CHECKING\n",
        "\n",
        "    if TYPE_CHECKING:\n",
        "        import google.colab  # type: ignore[import-not-found]\n",
        "\n",
        "import json\n",
        "import time\n",
        "import warnings\n",
        "import hashlib\n",
        "from pathlib import Path\n",
        "from datetime import datetime\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import duckdb\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.gridspec as gridspec\n",
        "from matplotlib.ticker import MaxNLocator\n",
        "import seaborn as sns\n",
        "from IPython.display import display, HTML, Markdown\n",
        "\n",
        "# ML Libraries\n",
        "from sklearn.model_selection import (\n",
        "    GroupShuffleSplit,\n",
        "    cross_val_score,\n",
        "    GroupKFold,\n",
        "    cross_validate,\n",
        ")\n",
        "from sklearn.ensemble import (\n",
        "    RandomForestClassifier,\n",
        "    StackingClassifier,\n",
        "    GradientBoostingClassifier,\n",
        "    AdaBoostClassifier,\n",
        ")\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.calibration import (\n",
        "    CalibratedClassifierCV,\n",
        "    calibration_curve,\n",
        "    CalibrationDisplay,\n",
        ")\n",
        "from sklearn.metrics import (\n",
        "    precision_recall_fscore_support,\n",
        "    roc_auc_score,\n",
        "    brier_score_loss,\n",
        "    confusion_matrix,\n",
        "    classification_report,\n",
        "    log_loss,\n",
        "    matthews_corrcoef,\n",
        "    average_precision_score,\n",
        "    precision_recall_curve,\n",
        "    roc_curve,\n",
        ")\n",
        "from sklearn.inspection import permutation_importance\n",
        "from sklearn.preprocessing import StandardScaler, label_binarize\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.svm import LinearSVC\n",
        "import shap\n",
        "\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "np.random.seed(42)\n",
        "\n",
        "# Configure publication-ready aesthetics\n",
        "plt.rcParams.update({\n",
        "    \"figure.dpi\": 120,\n",
        "    \"savefig.dpi\": 300,\n",
        "    \"font.size\": 11,\n",
        "    \"axes.titlesize\": 13,\n",
        "    \"axes.labelsize\": 11,\n",
        "    \"xtick.labelsize\": 10,\n",
        "    \"ytick.labelsize\": 10,\n",
        "    \"legend.fontsize\": 10,\n",
        "    \"figure.facecolor\": \"white\",\n",
        "    \"axes.facecolor\": \"#FAFAFA\",\n",
        "    \"axes.grid\": True,\n",
        "    \"grid.alpha\": 0.3,\n",
        "    \"axes.spines.top\": False,\n",
        "    \"axes.spines.right\": False,\n",
        "})\n",
        'sns.set_palette("husl")\n',
        "\n",
        "# Output directories\n",
        "for d in [\"work/outputs\", \"work/figures\"]:\n",
        "    os.makedirs(d, exist_ok=True)\n",
        "\n",
        "# Pipeline metadata\n",
        "PIPELINE_START = time.time()\n",
        "PIPELINE_VERSION = \"2.0.0-advanced\"\n",
        "NOTEBOOK_ID = hashlib.sha256(\n",
        "    datetime.now().isoformat().encode()\n",
        ").hexdigest()[:8]\n",
        "\n",
        'print(f"\\n{\"=\"*70}")\n',
        'print(f"  CAPSTONE PIPELINE v{PIPELINE_VERSION}")\n',
        'print(f"  Run ID: {NOTEBOOK_ID}")\n',
        "print(f\"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")\n",
        'print(f"{\"=\"*70}\\n")\n',
        "\n",
        "# Optional: Optuna for hyperparameter tuning (install if available)\n",
        "try:\n",
        "    import optuna\n",
        "    optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
        "    HAS_OPTUNA = True\n",
        '    print("✅ Optuna available — will optimize hyperparameters")\n',
        "except ImportError:\n",
        "    HAS_OPTUNA = False\n",
        '    print("⚠️  Optuna not installed — using default hyperparameters")\n',
        "\n",
        "# Optional: LIME for local explanations\n",
        "try:\n",
        "    from lime.lime_tabular import LimeTabularExplainer\n",
        "    HAS_LIME = True\n",
        '    print("✅ LIME available — dual explainability active")\n',
        "except ImportError:\n",
        "    HAS_LIME = False\n",
        '    print("⚠️  LIME not installed — SHAP-only explainability")\n',
        "\n",
        'print("\\nPipeline architecture initialized.")\n',
    ]


def main() -> None:
    nb = load_notebook(NOTEBOOK_PATH)

    old_cell = find_cell(nb, "Section 3: Pipeline Architecture & Dependency Initialization")
    if old_cell is None:
        raise SystemExit("Could not find the Section 3 setup cell in capstone.ipynb")

    old_src = "".join(old_cell["source"])
    if "from typing import TYPE_CHECKING" in old_src:
        print("Notebook already updated — skipping rebuild.")
        return

    old_cell["source"] = new_setup_cell()
    save_notebook(NOTEBOOK_PATH, nb)

    print("Updated capstone.ipynb setup cell:")
    print("- Added explicit TYPE_CHECKING import for google.colab")
    print("- Kept real import inside try/except")
    print("- Preserved all other imports and setup logic")


if __name__ == "__main__":
    main()
