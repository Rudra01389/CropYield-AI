"""Draw the architecture / workflow / ANN diagrams -> reports/diagrams/*.png.  Run: python make_diagrams.py"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import config as C

OUT = C.REPORT_DIR / "diagrams"
G_DARK, G_MID, G_LIGHT, GREY, AMBER, BLUE = "#1B5E20", "#2E7D32", "#E3F1E1", "#455A64", "#FFF3D6", "#E3F0FA"


def box(ax, x, y, w, h, text, fc=G_LIGHT, ec=G_MID, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#14231A", linespacing=1.3)


def arrow(ax, p, q, color=GREY, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.5, color=color,
                                 shrinkA=0, shrinkB=0))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(w / 2, h - 0.35, title, ha="center", fontsize=14, fontweight="bold", color=G_DARK)
    return fig, ax


def system_architecture():
    fig, ax = canvas(13, 7.2, "CropYield AI - System Architecture")
    for x, t in [(0.4, "1. Data sources"), (3.6, "2. Data pipeline"), (7.0, "3. Modelling"), (10.0, "4. Application")]:
        ax.text(x, 6.45, t, fontsize=11, fontweight="bold", color=G_MID)
    box(ax, 0.4, 4.7, 2.6, 1.4, "India Agriculture Crop\nProduction CSV\n(district yield, area)", BLUE, "#1E88E5")
    box(ax, 0.4, 2.9, 2.6, 1.4, "NASA POWER API\n(daily rain, temp, humidity)", BLUE, "#1E88E5")
    box(ax, 0.4, 1.0, 2.6, 1.4, "OpenStreetMap Nominatim\n(district coordinates)", BLUE, "#1E88E5")
    box(ax, 3.6, 4.7, 3.0, 1.4, "preprocess.py\nfilter Maharashtra / Rice / Kharif\nclean, flag, exclude + log", fs=8.5)
    box(ax, 3.6, 2.9, 3.0, 1.4, "fetch_coordinates.py\nfetch_weather.py (cached)\nJun-Sep aggregation", fs=8.5)
    box(ax, 3.6, 1.0, 3.0, 1.4, "data/processed/\nmerged + ann_ready_dataset.csv\n(642 rows, 34 districts)", AMBER, "#F9A825", 8.5)
    box(ax, 7.0, 1.0, 2.6, 1.4, "train_models.py\nchronological split\none-hot + StandardScaler\n(fit on train only)", fs=8.5)
    box(ax, 7.0, 2.9, 2.6, 1.4, "Models:\nA District-mean baseline\nB Linear Regression\nC ANN (Keras)", fs=8.5)
    box(ax, 7.0, 4.7, 2.6, 1.4, "models/ + reports/\n.keras, .joblib, metadata\nmetrics, plots, predictions", AMBER, "#F9A825", 8.5)
    box(ax, 10.0, 4.7, 2.6, 1.4, "src/predictor.py\nloads saved ANN +\npreprocessor, validates", fs=8.5)
    box(ax, 10.0, 2.9, 2.6, 1.4, "app.py (Streamlit)\nPredict | Data & trends |\nModel performance | About", fs=8.5, bold=True)
    box(ax, 10.0, 1.0, 2.6, 1.4, "User (browser)\nenters district, year, area,\nweather -> yield estimate", "#FFFFFF", GREY, 8.5)
    for p, q in [((3.0, 5.4), (3.6, 5.4)), ((3.0, 3.6), (3.6, 3.6)), ((3.0, 1.7), (3.6, 3.2)),
                 ((5.1, 4.7), (5.1, 4.3)), ((5.1, 2.9), (5.1, 2.4)),       # pipeline flows down
                 ((6.6, 1.7), (7.0, 1.7)),                                  # processed -> train
                 ((8.3, 2.4), (8.3, 2.9)), ((8.3, 4.3), (8.3, 4.7)),        # modelling flows up
                 ((9.6, 5.4), (10.0, 5.4)),                                 # artifacts -> predictor
                 ((11.3, 4.7), (11.3, 4.3)), ((11.3, 2.9), (11.3, 2.4))]:  # app flows down
        arrow(ax, p, q)
    fig.savefig(OUT / "01_system_architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def workflow():
    fig, ax = canvas(13, 6.0, "CropYield AI - Data to Dashboard Workflow")
    steps = [("Raw datasets\ncrop CSV +\nNASA POWER", BLUE), ("Cleaning &\nfiltering\nRice / Kharif", G_LIGHT),
             ("Merge crop +\nJun-Sep weather\n(district, year)", G_LIGHT),
             ("Chronological split\n1997-2015 train\n2016-17 val, 2018-19 test", AMBER),
             ("Encoding & scaling\none-hot district,\nStandardScaler\n(fit on train only)", G_LIGHT),
             ("ANN training\nAdam, MSE,\nearly stopping", G_LIGHT), ("Evaluation\nMAE, MSE,\nRMSE, R2", G_LIGHT),
             ("Saved model +\npreprocessor", AMBER), ("Prediction\n(inputs -> yield t/ha)", G_LIGHT),
             ("Streamlit\ndashboard", "#C8E6C9")]
    w, h, gap, x0 = 2.1, 1.5, 0.45, 0.45
    pos = []
    for i in range(10):
        col = i if i < 5 else 9 - i            # row 1 left->right, row 2 right->left (snake)
        pos.append((x0 + col * (w + gap), 3.5 if i < 5 else 1.4))
    for i, ((t, c), (x, y)) in enumerate(zip(steps, pos)):
        box(ax, x, y, w, h, t, c, G_MID, 8.5)
        ax.text(x + 0.12, y + h - 0.2, str(i + 1), fontsize=9, fontweight="bold", color=G_DARK)
    for i in range(9):
        (x, y), (nx, ny) = pos[i], pos[i + 1]
        if i == 4:
            arrow(ax, (x + w / 2, y), (nx + w / 2, ny + h))                         # down between rows
        elif i < 4:
            arrow(ax, (x + w, y + h / 2), (nx, ny + h / 2))                         # left -> right
        else:
            arrow(ax, (x, y + h / 2), (nx + w, ny + h / 2))                         # right -> left
    ax.text(6.5, 0.55, "Validation years are used for early stopping and seed selection only; "
            "test years are used once, for the final metrics.", ha="center", fontsize=9, style="italic", color=GREY)
    fig.savefig(OUT / "02_workflow.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def ann_architecture():
    import tensorflow as tf
    model = tf.keras.models.load_model(C.ROOT / "models" / "ann_model.keras", compile=False)
    n_in, params = model.input_shape[1], model.count_params()
    fig, ax = canvas(11, 7.0, "ANN Architecture (saved model)")
    layers = [(1.6, 8, f"Input layer\n{n_in} features\n5 numeric + 34 district\none-hot", "#1E88E5"),
              (4.6, 8, "Hidden layer 1\nDense 64\nReLU", G_MID), (7.3, 6, "Hidden layer 2\nDense 32\nReLU", G_MID),
              (9.9, 1, "Output layer\nDense 1, linear\nyield (t/ha)", "#F9A825")]
    cols = []
    for x, n, label, color in layers:
        ys = [3.3 + (i - (n - 1) / 2) * 0.46 for i in range(n)]
        cols.append((x, ys))
    for (x1, y1s), (x2, y2s) in zip(cols[:-1], cols[1:]):
        for ya in y1s:
            for yb in y2s:
                ax.plot([x1 + 0.17, x2 - 0.17], [ya, yb], color="#B0BEC5", lw=0.35, zorder=0)
    for (x, ys), (_, _, label, color) in zip(cols, layers):
        for y in ys:
            ax.add_patch(Circle((x, y), 0.17, fc=color, ec="white", lw=1))
        ax.text(x, 0.9, label, ha="center", va="center", fontsize=9, linespacing=1.35, color="#14231A")
    ax.text(5.5, 6.0, f"Total trainable parameters: {params:,}   |   Adam (lr 1e-3), MSE loss, batch 32, "
            "early stopping (patience 30)", ha="center", fontsize=9.5, color=G_DARK)
    ax.text(5.5, 5.62, "Only a subset of the neurons is drawn; layer sizes are as labelled.", ha="center",
            fontsize=8, style="italic", color=GREY)
    fig.savefig(OUT / "03_ann_architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ANN input features:", n_in, "parameters:", params)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    system_architecture()
    workflow()
    ann_architecture()
    print("Saved diagrams to", OUT)
