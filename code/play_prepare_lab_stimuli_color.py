# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools

# Set working directory
change_dir = "/Users/rezek_zhu/multimodal_attention"
os.chdir(change_dir)

save_dir = "data/lab_stimuli_color/picture"
os.makedirs(save_dir, exist_ok=True)

def generate_color_wheel(n=50):
    hsv = np.zeros((n, 3))
    hsv[:, 0] = np.linspace(0, 1, n, endpoint=False)
    hsv[:, 1] = 1
    hsv[:, 2] = 1
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)
    hex_colors = [mcolors.to_hex(c) for c in rgb]
    return hex_colors

EDGE_COLORS = generate_color_wheel(50)
COLOR_IDS = [f"c{i+1}" for i in range(len(EDGE_COLORS))]  # c1~c50

# Only one shape: Circle, textures as in the referenced play_prepare_lab_stimuli.py
SHAPES   = ["circle"]
TEXTURES = {
    "cross": "x"
}

def draw_stimulus(shape, texture, edge_color, color_id, save_dir):
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Use hatch/facecolor method to mimic the referenced code
    hatch_pattern = TEXTURES[texture]
    patch = Circle((0, 0), 0.5, facecolor="white", edgecolor=edge_color, hatch=hatch_pattern, lw=4, joinstyle='miter')
    ax.add_patch(patch)

    # filename: shape_texture_colorID.png, e.g., circle_slash_c23.png
    fname = f"{shape}_{texture}_{color_id}.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {fpath}")

# Generate for all combinations
for texture in TEXTURES:
    for color_id, edge_color in zip(COLOR_IDS, EDGE_COLORS):
        draw_stimulus("circle", texture, edge_color, color_id, save_dir)

print("✅ All color stimuli (circle + texture, color as c1~c50) generated.")

# ---- Draw and annotate the color wheel with selected colors ----

import matplotlib.colors as mcolors

n_colors = len(EDGE_COLORS)
theta = np.linspace(0, 2*np.pi, n_colors, endpoint=False)
r = 1.0

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')

# Draw colored dots (or lines) around color wheel
for i, (angle, color) in enumerate(zip(theta, EDGE_COLORS)):
    ax.plot([angle], [r], marker='o', markersize=20, color=color, markeredgecolor='k')
    # annotate with color id (c1~c50)
    ax.text(angle, r+0.1, f"c{i+1}", color='black', fontsize=10, ha='center', va='center')

ax.set_rticks([])
ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)
ax.set_ylim(0, r+0.2)
plt.title("Color Wheel with Selected Edge Colors (c1~c50)", pad=24)
plt.tight_layout()
plt.show()
# %%
