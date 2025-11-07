# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import itertools

# Set working directory
change_dir = "/Users/rezek_zhu/multimodal_attention"
os.chdir(change_dir)
os.makedirs("data/lab_stimuli", exist_ok=True)

# %%
# Define parameter spaces
SHAPES = ["circle", "square", "triangle"]
TEXTURES = {
    "dot": ".",        # dotted pattern
    "dash": "-",       # horizontal lines
    "cross": "+",      # crosshatch
}
EDGE_COLORS = ["blue", "red", "green"]

# %%
def draw_stimulus(shape, texture, edge_color, save_dir="data/lab_stimuli"):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    hatch_pattern = TEXTURES[texture]
    common_kwargs = dict(
        facecolor="white", edgecolor=edge_color, hatch=hatch_pattern, lw=4
    )

    if shape == "circle":
        patch = Circle((0, 0), 0.5, **common_kwargs)
    elif shape == "square":
        patch = Rectangle((-0.5, -0.5), 1, 1, **common_kwargs)
    elif shape == "triangle":
        patch = Polygon([[0, 0.6], [-0.6, -0.5], [0.6, -0.5]], **common_kwargs)
    else:
        raise ValueError("Unknown shape")

    ax.add_patch(patch)

    fname = f"{shape}_{texture}_{edge_color}.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {fpath}")

# %%
# Generate all combinations
for shape, texture, color in itertools.product(SHAPES, TEXTURES.keys(), EDGE_COLORS):
    draw_stimulus(shape, texture, color)

print("✅ All stimuli generated.")
# %%
