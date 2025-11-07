# %% objective
# Cluster images using embeddings from CLIP model
# %% preparation

import os
import numpy as np
from PIL import Image

# Set working directory to code directory
os.chdir('/Users/rezek_zhu/multimodal_attention')
print(os.getcwd())


focus_type = 'lab_stimuli'
if focus_type == 'video':
    # Change these paths to your embedding and picture directory as needed
    embedding_path = 'data/video/embedding/openclip/ViT-H-14-378-quickgelu/dfn5b/video_embedding.npy'
    picture_dir = 'data/video/frame/40'
    video_dir = 'data/video/original_clip'

    picture_embedding = np.load(embedding_path)

    # Load video names to ensure correct correspondence
    video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    # Convert video names to corresponding picture names
    def video_to_picture_name(video_name):
        """Convert video name (e.g., 'video.mp4') to picture name (e.g., 'video@40.png')"""
        return video_name.replace('.mp4', '@40.png')

    picture_name_list = np.array([video_to_picture_name(v) for v in video_name_list])

    print("Loaded embedding shape:", picture_embedding.shape)
    print("Loaded names (from picture_dir):", len(picture_name_list))
else:
    embedding_path = 'data/lab_stimuli_v2/embedding/openclip/ViT-H-14-378-quickgelu/dfn5b/picture_embedding.npy'
    picture_dir = 'data/lab_stimuli_v2/picture'
    picture_embedding = np.load(embedding_path)
    picture_name_list = [f for f in os.listdir(picture_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    picture_name_list = np.array(picture_name_list)
    print("Loaded embedding shape:", picture_embedding.shape)
    print("Loaded names (from picture_dir):", len(picture_name_list))

# %% helper
# clustering
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

def get_image(path):
    # Read an image as np.array (RGB)
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print("Error loading image:", path, e)
        return None

def analyze_clusters(embedding, name_list, picture_dir, cluster_num=11):
    # Cluster color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"
    ]

    embedding_np = embedding
    # 1-1. PCA: using optimal n_components that exceeds 95% explained variance ratio
    pca_full = PCA(random_state=42)
    pca_full.fit(embedding_np)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    pca_threshold = 0.95
    best_n_components = np.argmax(cumsum >= pca_threshold) + 1

    pca = PCA(n_components=best_n_components, random_state=42)
    reduced_embedding = pca.fit_transform(embedding_np)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             cumsum, 'bo-')
    plt.axhline(y=pca_threshold, color='grey', linestyle='--')
    plt.axvline(x=best_n_components, color='grey', linestyle='--')
    plt.title(f'Cumulative Explained Variance Ratio\n(Selected {best_n_components} components)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    # 1-2. Hierarchical clustering: choose optimal k with silhouette
    dist_matrix = pdist(reduced_embedding)
    linkage_matrix = hierarchy.linkage(dist_matrix, method='ward')

    best_score = -1
    optimal_k = None
    for k in range(3, 15):
        labels_k = hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(reduced_embedding, labels_k)
        if score > best_score:
            best_score = score
            optimal_k = k
    #optimal_k = cluster_num

    # 1-3. Clustering
    dist_matrix = pdist(reduced_embedding)
    linkage_matrix = hierarchy.linkage(dist_matrix, method='ward')
    cluster_labels = hierarchy.fcluster(linkage_matrix, optimal_k, criterion='maxclust')

    # Plot 1: dendrogram
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(linkage_matrix, truncate_mode='lastp', p=optimal_k,
                         leaf_font_size=15,
                         link_color_func=lambda k: 'black')
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        label.set_color(colors[i])
    plt.title(f'Hierarchical Clustering Dendrogram (k={optimal_k})', fontsize=20)
    plt.show()

    # Plot 2: 2D scatter plot with representative images
    pca_2d = PCA(n_components=2, random_state=42)
    embedding_2d = pca_2d.fit_transform(reduced_embedding)

    plt.figure(figsize=(12, 8))
    for cluster_id in range(1, optimal_k + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        plt.scatter(embedding_2d[cluster_indices, 0], embedding_2d[cluster_indices, 1],
                    color=colors[cluster_id-1], label=f'Cluster {cluster_id}')

    # Add representative images
    for cluster_id in range(1, optimal_k + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = np.mean(embedding_2d[cluster_indices], axis=0)
        distances = np.linalg.norm(embedding_2d[cluster_indices] - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]

        img_path = os.path.join(picture_dir, name_list[closest_idx])
        frame = get_image(img_path)
        if frame is not None:
            frame_pil = Image.fromarray(frame)
            frame_pil.thumbnail((50, 50))
            imagebox = OffsetImage(frame_pil, zoom=1.0)
            ab = AnnotationBbox(imagebox, cluster_center, frameon=True,
                                bboxprops=dict(edgecolor=colors[cluster_id-1],
                                               facecolor=colors[cluster_id-1],
                                               linewidth=3))
            plt.gca().add_artist(ab)
    plt.show()

    # Plot 3: most and least representative images for each cluster
    plt.figure(figsize=(15, 4 * optimal_k))
    plt.suptitle('Most/Least Representative Images Per Cluster', fontsize=30, y=1.02)
    for cluster_id in range(1, optimal_k + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_center = np.mean(embedding_2d[cluster_indices], axis=0)
        distances = np.linalg.norm(embedding_2d[cluster_indices] - cluster_center, axis=1)

        closest_indices = cluster_indices[np.argsort(distances)[:2]]  # 2 most representative
        farthest_indices = cluster_indices[np.argsort(distances)[-2:]]  # 2 least representative

        for j, (idx, rep_type) in enumerate([(i, "Most") for i in closest_indices] +
                                            [(i, "Least") for i in farthest_indices]):
            img_path = os.path.join(picture_dir, name_list[idx])
            frame = get_image(img_path)
            if frame is not None:
                ax = plt.subplot(optimal_k, 4, (cluster_id - 1) * 4 + j + 1)
                plt.imshow(cv2.resize(frame, (300, 300)))
                plt.axis('off')
                plt.title(f'Cluster {cluster_id}\n{rep_type} #{j%2 + 1}',
                          color=colors[cluster_id-1],
                          fontweight='bold')
    plt.tight_layout()
    plt.show()

# lab stimuli similarity and correlation
def plot_lab_stimuli_similarity_and_correlation(picture_embedding, picture_name_list):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    # 1. Compute embedding cosine similarity matrix
    embedding_matrix = np.array(picture_embedding)
    cos_sim_matrix = cosine_similarity(embedding_matrix)

    # 2. Parse titles into attributes for each image
    def parse_attributes(name):
        # Example: 'circle_cross_red'
        # Adjust split logic as necessary based on exact name format.
        parts = name.split('_')
        # Here assuming [shape, texture, color]
        shape, texture, color = parts
        return shape, texture, color

    shapes = []
    textures = []
    colors = []
    for name in picture_name_list:
        shape, texture, color = parse_attributes(name)
        shapes.append(shape)
        textures.append(texture)
        colors.append(color)

    n_images = len(picture_name_list)

    # 3. Create similarity matrices for the 3 label dimensions
    def similarity_matrix(attr_list):
        m = np.zeros((n_images, n_images))
        for i in range(n_images):
            for j in range(n_images):
                m[i, j] = 1 if attr_list[i] == attr_list[j] else 0
        return m

    shape_matrix = similarity_matrix(shapes)
    texture_matrix = similarity_matrix(textures)
    color_matrix = similarity_matrix(colors)

    # 4. Plot the matrices
    matrices = [
        (cos_sim_matrix, 'Embedding Cosine Similarity'),
        (shape_matrix, 'Shape Similarity (1=same)'),
        (texture_matrix, 'Texture Similarity (1=same)'),
        (color_matrix, 'Color Similarity (1=same)')
    ]

    plt.figure(figsize=(24, 6))
    for idx, (matrix, title) in enumerate(matrices):
        plt.subplot(1, 4, idx + 1)
        im = plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.title(title, fontsize=14)
        plt.xlabel('Image Index')
        plt.ylabel('Image Index')
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # 5. Compute and plot correlation heatmap
    def get_upper_triangle(matrix):
        # Returns upper triangle vector (excluding diagonal)
        return matrix[np.triu_indices_from(matrix, k=1)]

    upper_data = []
    matrix_labels = []
    for matrix, title in matrices:
        upper_vec = get_upper_triangle(matrix)
        upper_data.append(upper_vec)
        matrix_labels.append(title)

    upper_data_array = np.vstack(upper_data)  # shape: (4, N) where N = n_images*(n_images-1)//2

    corr_matrix = np.corrcoef(upper_data_array)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=matrix_labels,
        yticklabels=matrix_labels,
        cmap='coolwarm'
    )
    plt.title('Correlation of Upper Triangle Vectors Between Similarity Matrices')
    plt.tight_layout()
    plt.show()

# %% for whole set
# clustering and visualization
analyze_clusters(picture_embedding, picture_name_list, picture_dir)

# similarity and correlation
plot_lab_stimuli_similarity_and_correlation(picture_embedding, picture_name_list)

# %% subset attention
import numpy as np

def filter_stimuli(names, embeddings, shape=None, texture=None, color=None):
    """
    names: list/array of image file names
    embeddings: numpy array or tensor, shape (N, ...)
    shape, texture, color: str or list/tuple of strings or None
    Returns: filtered_names, filtered_embeds, selected_indices
    """
    if isinstance(names, np.ndarray):
        names = names.tolist()

    if shape is not None and not isinstance(shape, (list, tuple, set)):
        shape = [shape]
    if texture is not None and not isinstance(texture, (list, tuple, set)):
        texture = [texture]
    if color is not None and not isinstance(color, (list, tuple, set)):
        color = [color]
        
    selected_indices = []
    for idx, fname in enumerate(names):
        # file names like: shape_texture_color.png
        parts = fname.rsplit('.', 1)[0].split('_')
        if len(parts) < 3:
            continue
        match = True
        if shape is not None and parts[0] not in shape:
            match = False
        if texture is not None and parts[1] not in texture:
            match = False
        if color is not None and parts[2] not in color:
            match = False
        if match:
            selected_indices.append(idx)
    filtered_names = [names[i] for i in selected_indices]
    if hasattr(embeddings, "__getitem__"):
        filtered_embeds = embeddings[selected_indices]
    else:
        filtered_embeds = [embeddings[i] for i in selected_indices]

    print(
        f"Selected {len(filtered_names)} images",
        f"{'with shape in ' + str(shape) if shape else ''}",
        f"{'with texture in ' + str(texture) if texture else ''}",
        f"{'with color in ' + str(color) if color else ''}"
    )
    return filtered_names, filtered_embeds, selected_indices


filtered_names, filtered_embeds, selected_indices = filter_stimuli(
    picture_name_list, picture_embedding,
    texture=["cross","slash"],
    shape=["circle", "square"],
    color=["blue", "red", "green"]
)

# clustering and visualization
analyze_clusters(filtered_embeds, filtered_names, picture_dir, cluster_num=3)

# similarity and correlation
plot_lab_stimuli_similarity_and_correlation(filtered_embeds, filtered_names)
# %%
