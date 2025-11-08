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
SHAPES   = ["circle", "square", "triangle", "star", "cross"]  # 改"snake"为"cross"
TEXTURES = {
    "slash": "/", 
    "cross": "x", 
    "dot": ".", 
    "dash": "-", 
    "star": "*"
} # rare: "*(starry hatch)
EDGE_COLORS   = ["blue", "red", "green", "#71F3F5", "#E24DAB"]          # rare: "#71F3F5", "#E24DAB"

# %%
from matplotlib.path import Path
import numpy as np

def regular_polygon_vertices(center, radius, num_vertices, rotation=0):
    # Returns coordinates of regular polygon
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + rotation
    return np.stack([center[0] + radius * np.cos(angles),
                     center[1] + radius * np.sin(angles)], axis=1)

def star_vertices(center, outer_radius, inner_radius, num_points):
    # Returns coordinates of a star shape
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    radii = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(num_points * 2)])
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def cross_vertices(center, size=0.6, thickness=0.2):
    """
    Returns the vertices for a cross shape (十字) as a single continuous (non-intersecting) Polygon contour.
    The cross is centered at 'center', of total length 'size', and bar thickness 'thickness'.
    The path starts at the top edge of the vertical bar and traces the entire outer contour clockwise.
    """
    cx, cy = center
    s = size / 2
    t = thickness / 2

    # Define the 12 outer points of the cross, clockwise from top
    verts = [
        (cx - t, cy + s),                      # 1: left top
        (cx + t, cy + s),                      # 2: right top
        (cx + t, cy + t),                      # 3: right stem/top, inside corner
        (cx + s, cy + t),                      # 4: right far end (horizontal bar, upper)
        (cx + s, cy - t),                      # 5: right far end (horizontal bar, lower)
        (cx + t, cy - t),                      # 6: right stem/bottom, inside corner
        (cx + t, cy - s),                      # 7: right bottom
        (cx - t, cy - s),                      # 8: left bottom
        (cx - t, cy - t),                      # 9: left stem/bottom, inside corner
        (cx - s, cy - t),                      # 10: left far end (horizontal bar, lower)
        (cx - s, cy + t),                      # 11: left far end (horizontal bar, upper)
        (cx - t, cy + t)                       # 12: left stem/top, inside corner
        # (back to 1 on close)
    ]
    return np.array(verts)

def draw_stimulus(shape, texture, edge_color, save_dir="data/lab_stimuli_v4/picture"):
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
        ax.add_patch(patch)
    elif shape == "square":
        patch = Rectangle((-0.5, -0.5), 1, 1, **common_kwargs)
        ax.add_patch(patch)
    elif shape == "triangle":
        verts = regular_polygon_vertices(center=(0,0), radius=0.6, num_vertices=3, rotation=np.pi/2)
        patch = Polygon(verts, closed=True, **common_kwargs)
        ax.add_patch(patch)
    elif shape == "pentagon":
        verts = regular_polygon_vertices(center=(0,0), radius=0.6, num_vertices=5, rotation=np.pi/2)
        patch = Polygon(verts, closed=True, **common_kwargs)
        ax.add_patch(patch)
    elif shape == "star":
        verts = star_vertices(center=(0,0), outer_radius=0.6, inner_radius=0.25, num_points=5)
        patch = Polygon(verts, closed=True, **common_kwargs)
        ax.add_patch(patch)
    elif shape == "cross":
        verts = cross_vertices(center=(0,0), size=1.0, thickness=0.28)
        patch = Polygon(verts, closed=True, **common_kwargs)
        ax.add_patch(patch)
    else:
        raise ValueError("Unknown shape")

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