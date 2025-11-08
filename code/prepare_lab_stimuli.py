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
SHAPES   = ["circle", "square", "triangle", "star", "snake"]  # rare: "star", "snake"
TEXTURES = {
    "slash": "/", 
    "cross": "x", 
    "dot": ".", 
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

def snake_vertices(center, length=1.2, amplitude=0.22, segments=60, rotations=2):
    """
    Make a 'snake-like' wavy path centered at 'center'.
    'rotations': how many sinusoidal waves in the path.
    """
    t = np.linspace(-length/2, length/2, segments)
    x = t
    y = amplitude * np.sin(2 * np.pi * rotations * t / length)
    # Optionally, add slight randomness for more organic shape:
    np.random.seed(42)
    y += np.random.normal(scale=0.03, size=y.shape)
    # Centering:
    x += center[0]
    y += center[1]
    return np.stack([x, y], axis=1)

def draw_stimulus(shape, texture, edge_color, save_dir="data/lab_stimuli_v3/picture"):
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
    elif shape == "snake":
        verts = snake_vertices(center=(0,0), length=1.2, amplitude=0.22, segments=80, rotations=3)
        # snake uses Polygon for thick closed shape? No, use Line2D for wiggly line:
        import matplotlib.lines as mlines
        # Because we want texture/hatch, fake with a buffer region:
        snake_linewidth = 16
        ax.plot(verts[:,0], verts[:,1],
                color=edge_color, lw=snake_linewidth, solid_capstyle='round', zorder=1)
        # Texture overlay hack: draw same line on top with white line and hatch lines if possible
        # Overlay with a slightly thinner line for hatch transparency
        ax.plot(verts[:,0], verts[:,1],
                color='white', lw=snake_linewidth-6, alpha=0.9, zorder=2)
        # Try to fake hatch: overlay "texture" pattern: dense short dashes with color
        if hatch_pattern == '/':
            ax.plot(verts[:,0], verts[:,1],
                    color='none', lw=1) # do nothing
        elif hatch_pattern in ('.','x','O','*'):
            # Place small dots, crosses, stars etc. along the line
            n_texture = 18
            pos_idx = np.linspace(0, verts.shape[0]-1, n_texture, dtype=int)
            for i in pos_idx:
                if hatch_pattern == '.':
                    ax.plot(verts[i,0], verts[i,1],'o', color=edge_color, markersize=6, alpha=0.8, zorder=3)
                elif hatch_pattern == 'O':
                    ax.plot(verts[i,0], verts[i,1],'o', markerfacecolor='white', markeredgecolor=edge_color, markersize=8, alpha=0.8, zorder=3)
                elif hatch_pattern == 'x':
                    ax.plot(verts[i,0], verts[i,1],'x', color=edge_color, markersize=7, zorder=3)
                elif hatch_pattern == '*':
                    ax.plot(verts[i,0], verts[i,1],'*', color=edge_color, markersize=10, alpha=0.7, zorder=3)
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