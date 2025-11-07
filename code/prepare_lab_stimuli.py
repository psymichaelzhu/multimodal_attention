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
# Define parameter spaces for stimuli
# v1: simple shapes
SHAPES = ["circle", "square", "triangle"]
TEXTURES = {
    "dot": ".",        # dotted pattern
    "dash": "-",       # horizontal lines
    "cross": "+",      # crosshatch
}
EDGE_COLORS = ["blue", "red", "green"]

# v2: add rare/complex shapes and colors
# Instead of a classic star, use a highly irregular, "weird star"-shaped polygon as a rare shape.
SHAPES = ["circle", "square", "triangle", "pentagon", "weirdstar"]  # 'weirdstar' is a rare, odd shape.
TEXTURES = {
    "slash": "/", 
    "circle": "O", 
    "cross": "x", 
    "dot": ".", 
    "star": "*"
}  # 'star': star-shaped hatch pattern as rare
EDGE_COLORS = ["blue", "red", "green", "pink", "chartreuse"]  # 'chartreuse': rare color

# %%
from matplotlib.path import Path
import numpy as np

def regular_polygon_vertices(center, radius, num_vertices, rotation=0):
    """Return coordinates for a regular polygon centered at 'center'."""
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + rotation
    return np.stack([center[0] + radius * np.cos(angles),
                     center[1] + radius * np.sin(angles)], axis=1)

def weird_star_vertices(center, factor=0.7):
    """
    Generate a highly irregular, wavy-edged 'weirdstar' shape.
    Produces a 'starburst' like shape with randomized radius perturbations for each point.
    """
    num_points = 16
    base_radius = 0.6
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Create radially varying noise for uneven/weird outline
    np.random.seed(42)
    radii_noise = (np.sin(angles * 7) + 1.5) * 0.25 + (np.random.rand(num_points) - 0.5) * 0.14
    radii = base_radius * (1 + factor * radii_noise)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def draw_stimulus(shape, texture, edge_color, save_dir="data/lab_stimuli_v2/picture"):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    hatch_pattern = TEXTURES[texture]
    common_kwargs = dict(
        facecolor="white", edgecolor=edge_color, hatch=hatch_pattern, lw=4,
        joinstyle='miter'
    )

    if shape == "circle":
        patch = Circle((0, 0), 0.5, **common_kwargs)
    elif shape == "square":
        patch = Rectangle((-0.5, -0.5), 1, 1, **common_kwargs)
    elif shape == "triangle":
        verts = regular_polygon_vertices(center=(0, 0), radius=0.6, num_vertices=3, rotation=np.pi/2)
        patch = Polygon(verts, closed=True, **common_kwargs)
    elif shape == "pentagon":
        verts = regular_polygon_vertices(center=(0, 0), radius=0.6, num_vertices=5, rotation=np.pi/2)
        patch = Polygon(verts, closed=True, **common_kwargs)
    elif shape == "weirdstar":
        verts = weird_star_vertices(center=(0, 0), factor=0.7)
        patch = Polygon(verts, closed=True, **common_kwargs)
    else:
        raise ValueError("Unknown shape: " + str(shape))

    ax.add_patch(patch)

    fname = f"{shape}_{texture}_{edge_color}.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {fpath}")

# %%
# Generate all combinations of shape, texture, and color
for shape, texture, color in itertools.product(SHAPES, TEXTURES.keys(), EDGE_COLORS):
    draw_stimulus(shape, texture, color)

print("All stimuli generated.")
# %%