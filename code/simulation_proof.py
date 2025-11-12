# %% objective: prove that the norm of the difference of two vectors in a hemisphere is equal to the norm of the difference of the two vectors in a plane
# %% preparation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %% helper function
def simulate_vector_combination(a1, b1, c):
    """
    Simulate vector combination in a hemisphere.
    
    Given:
    - a1, b1: two vectors in the circular cross-section
    - c: a vector inside the hemisphere
    
    Compute:
    - a2 = c + a1
    - b2 = c + b1
    - d = a2 - b2
    
    Args:
        a1: numpy array of shape (3,) representing vector a1
        b1: numpy array of shape (3,) representing vector b1
        c: numpy array of shape (3,) representing vector c
    
    Returns:
        tuple: (norm_d, cos_similarity)
            - norm_d: float, norm of d = a2 - b2
            - cos_similarity: float, cosine similarity between a2 and b2
    """
    # Convert to numpy arrays if not already
    a1 = np.array(a1)
    b1 = np.array(b1)
    c = np.array(c)
    
    # Compute resulting vectors
    a2 = c + a1
    b2 = c + b1
    print(f"a2 = {a2}")
    print(f"b2 = {b2}")
    
    # Compute difference
    d = a2 - b2
    # 1. Compute norm of d
    norm_d = np.linalg.norm(d)
    
    # 2. Compute cosine similarity between a2 and b2
    norm_a2 = np.linalg.norm(a2)
    norm_b2 = np.linalg.norm(b2)
    
    if norm_a2 == 0 or norm_b2 == 0:
        cos_similarity = 0.0
        angle_rad = 0.0
    else:
        cos_similarity = np.dot(a2, b2) / (norm_a2 * norm_b2)
        # Compute angle in radians
        angle_rad = np.arccos(np.clip(cos_similarity, -1.0, 1.0))
    
    return norm_d, cos_similarity, angle_rad


def visualize_vector_combination(a1, b1, c, save_path=None):
    """
    Visualize all vectors involved in the vector combination.
    
    Args:
        a1: numpy array of shape (3,) representing vector a1
        b1: numpy array of shape (3,) representing vector b1
        c: numpy array of shape (3,) representing vector c
        save_path: str, optional path to save the figure
    """
    # Convert to numpy arrays if not already
    a1 = np.array(a1)
    b1 = np.array(b1)
    c = np.array(c)
    
    # Compute resulting vectors
    a2 = c + a1
    b2 = c + b1
    d = a2 - b2
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin
    origin = np.array([0, 0, 0])
    
    # Plot vectors from origin
    # a1 (blue)
    ax.quiver(origin[0], origin[1], origin[2], 
              a1[0], a1[1], a1[2], 
              color='blue', arrow_length_ratio=0.15, linewidth=2.5, label='a1')
    
    # b1 (blue)
    ax.quiver(origin[0], origin[1], origin[2], 
              b1[0], b1[1], b1[2], 
              color='blue', arrow_length_ratio=0.15, linewidth=2.5, label='b1')
    
    # c (orange)
    ax.quiver(origin[0], origin[1], origin[2], 
              c[0], c[1], c[2], 
              color='orange', arrow_length_ratio=0.15, linewidth=2.5, label='c')
    
    # a2 = c + a1 (green)
    ax.quiver(origin[0], origin[1], origin[2], 
              a2[0], a2[1], a2[2], 
              color='green', arrow_length_ratio=0.15, linewidth=2.5, label='a2 = c + a1')
    
    # b2 = c + b1 (green)
    ax.quiver(origin[0], origin[1], origin[2], 
              b2[0], b2[1], b2[2], 
              color='green', arrow_length_ratio=0.15, linewidth=2.5, label='b2 = c + b1')
    
    # a1 - b1 (black, no arrow, solid line from b1 to a1)
    ax.plot([b1[0], a1[0]], [b1[1], a1[1]], [b1[2], a1[2]], 
            color='black', linewidth=2.5, linestyle='-', label='a1 - b1')
    
    # a2 - b2 (black, no arrow, solid line from b2 to a2)
    ax.plot([b2[0], a2[0]], [b2[1], a2[1]], [b2[2], a2[2]], 
            color='black', linewidth=2.5, linestyle='-', label='a2 - b2')
    
    # c to b2 (light gray, no arrow, dashed line)
    ax.plot([c[0], b2[0]], [c[1], b2[1]], [c[2], b2[2]], 
            color='gray', linewidth=1.5, linestyle='--', alpha=0.7, label='c to b2')
    
    # c to a2 (light gray, no arrow, dashed line)
    ax.plot([c[0], a2[0]], [c[1], a2[1]], [c[2], a2[2]], 
            color='gray', linewidth=1.5, linestyle='--', alpha=0.7, label='c to a2')
    
    # a1 to a2 (light gray, no arrow, dashed line)
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], [a1[2], a2[2]], 
            color='gray', linewidth=1.5, linestyle='--', alpha=0.7, label='a1 to a2')
    
    # b1 to b2 (light gray, no arrow, dashed line)
    ax.plot([b1[0], b2[0]], [b1[1], b2[1]], [b1[2], b2[2]], 
            color='lightgray', linewidth=1.5, linestyle='--', alpha=0.7, label='b1 to b2')
    
    # Add labels at vector endpoints
    ax.text(a1[0], a1[1], a1[2], 'a1', fontsize=10, color='blue')
    ax.text(b1[0], b1[1], b1[2], 'b1', fontsize=10, color='blue')
    ax.text(c[0], c[1], c[2], 'c', fontsize=10, color='orange')
    ax.text(a2[0], a2[1], a2[2], 'a2', fontsize=10, color='green')
    ax.text(b2[0], b2[1], b2[2], 'b2', fontsize=10, color='green')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Combination Visualization\na2 = c + a1, b2 = c + b1, d = a2 - b2')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=9)
    
    # Set equal aspect ratio
    max_range = np.array([a1, b1, c, a2, b2]).max()
    ax.set_xlim([-max_range*0.2, max_range*1.2])
    ax.set_ylim([-max_range*0.2, max_range*1.2])
    ax.set_zlim([-max_range*0.2, max_range*1.2])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# %%
# Example vectors
a1 = np.array([1.0, 0.0, 0.0])
b1 = np.array([0.0, 1.0, 0.0])

# Try 10 different c vectors in the upper space
for i in range(10):
    # Generate random c vector in upper space (positive z component)
    c = np.random.randn(3)
    c[2] = abs(c[2])  # Ensure z component is positive (upper space)
    
    print(f"\n{'='*60}")
    print(f"Test {i+1}/10: c = [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
    print('='*60)
    
    norm_d, cos_sim, angle_rad = simulate_vector_combination(a1, b1, c)
    
    print("Simulation Results:")
    print(f"a1 = {a1}")
    print(f"b1 = {b1}")
    print(f"c = {c}")
    print(f"\nNorm of d (a2 - b2): {norm_d:.4f}")
    print(f"Cosine similarity between a2 and b2: {cos_sim:.4f}")
    print(f"angle between a2 and b2: {angle_rad:.4f} radians")

    # Note: d = a2 - b2 = (c + a1) - (c + b1) = a1 - b1
    # So norm(d) should equal norm(a1 - b1)
    print(f"\nVerification: norm(a1 - b1) = {np.linalg.norm(a1 - b1):.4f}")
    print(f"cosine similarity between a1 and b1: {np.dot(a1, b1) / (np.linalg.norm(a1) * np.linalg.norm(b1)):.4f}")
    print(f"angle between a1 and b1: {np.arccos(np.clip(np.dot(a1, b1) / (np.linalg.norm(a1) * np.linalg.norm(b1)), -1.0, 1.0)):.4f} radians")
    
    # Visualize the vectors
    print("\nGenerating visualization...")
    visualize_vector_combination(a1, b1, c, save_path='vector_combination_visualization.png')

# %%
