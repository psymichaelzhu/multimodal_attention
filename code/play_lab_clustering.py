# %% objective
# Cluster images using embeddings from CLIP model

# %% preparation    
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Load model (CLIP or OpenCLIP, using open_clip for broad compatibility) ---------
try:
    import open_clip
    model_name = 'ViT-H-14-378-quickgelu'
    pretrained_name = 'dfn5b'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
    tokenizer = open_clip.get_tokenizer(model_name)
except ImportError:
    import clip
    model_name = 'ViT-B/32'
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize

model = model.to(device)
model.eval()


# %% --------- Specify your picture directory ---------
os.chdir('/Users/rezek_zhu/multimodal_attention')  
print(os.getcwd())
target_dir = 'data/genAI_stimuli2'
picture_dir = os.path.join(target_dir, 'picture')  # <-- EDIT this path as needed

# %% --------- Collect list of image paths ---------
img_filenames = [
    f for f in os.listdir(picture_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
]
img_paths = [os.path.join(picture_dir, f) for f in img_filenames]

# --------- Load images ---------
images = []
valid_names = []
for path, name in zip(img_paths, img_filenames):
    try:
        img = Image.open(path).convert('RGB')
        images.append(img)
        valid_names.append(name)
    except Exception as e:
        print(f"Failed to load {name}: {e}")

print(f"Loaded {len(images)} images.")

# %% --------- Compute or load CLIP embeddings and ensure correspondence ---------
embedding_dir = os.path.join(target_dir, 'embedding')
os.makedirs(embedding_dir, exist_ok=True)
embedding_path = os.path.join(embedding_dir, 'picture_embedding.npy')
name_path = os.path.join(embedding_dir, 'picture_name_list.npy')
overwrite = False

def compute_embeddings(images, preprocess, model, device, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch_imgs = images[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            batch_emb = model.encode_image(batch_tensors)
            all_embeddings.append(batch_emb.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()

need_compute = True

if os.path.exists(embedding_path) and os.path.exists(name_path) and not overwrite:
    print("Loading existing embeddings ...")
    loaded_embeddings = np.load(embedding_path)
    loaded_names = list(np.load(name_path, allow_pickle=True))

    # Validate correspondence with current images
    loaded_name_set = set(loaded_names)
    valid_name_set = set(valid_names)
    if loaded_name_set == valid_name_set and len(loaded_names) == len(valid_names):
        # Match order
        name_to_idx = {name: idx for idx, name in enumerate(loaded_names)}
        indices = [name_to_idx[name] for name in valid_names]
        embeddings = loaded_embeddings[indices]
        print(f"Loaded embeddings shape: {embeddings.shape} and matched against current image list.")
        need_compute = False

    else:
        # Names do not perfectly match: delete extras and compute missing
        print("Embeddings and image names do not match perfectly.")
        # Find extra/obsolete embeddings to remove and missing images to compute
        names_intersection = valid_name_set & loaded_name_set
        names_only_in_valid = list(valid_name_set - loaded_name_set)
        names_only_in_loaded = list(loaded_name_set - valid_name_set)

        # Gather embeddings for names in intersection, preserving order with valid_names
        if len(names_intersection) > 0:
            name_to_idx_loaded = {name: idx for idx, name in enumerate(loaded_names)}
            intersection_indices = [name_to_idx_loaded[name] for name in valid_names if name in names_intersection]
            kept_embeddings = loaded_embeddings[intersection_indices]
            kept_names = [name for name in valid_names if name in names_intersection]
        else:
            kept_embeddings = np.zeros((0, loaded_embeddings.shape[1])) if loaded_embeddings.ndim == 2 else np.zeros((0,))
            kept_names = []

        # Compute embeddings for missing images
        missing_img_indices = [i for i, name in enumerate(valid_names) if name in names_only_in_valid]
        if missing_img_indices:
            print(f"Computing embeddings for {len(missing_img_indices)} missing images ...")
            missing_images = [images[i] for i in missing_img_indices]
            missing_embeddings = compute_embeddings(missing_images, preprocess, model, device)
        else:
            missing_embeddings = np.zeros((0, loaded_embeddings.shape[1])) if loaded_embeddings.ndim == 2 else np.zeros((0,))

        # Combine
        if len(kept_embeddings) > 0 and len(missing_embeddings) > 0:
            embeddings = np.concatenate([kept_embeddings, missing_embeddings], axis=0)
            valid_names_new = kept_names + [valid_names[i] for i in missing_img_indices]
        elif len(kept_embeddings) > 0:
            embeddings = kept_embeddings
            valid_names_new = kept_names
        elif len(missing_embeddings) > 0:
            embeddings = missing_embeddings
            valid_names_new = [valid_names[i] for i in missing_img_indices]
        else:
            embeddings = np.zeros((0, loaded_embeddings.shape[1])) if loaded_embeddings.ndim == 2 else np.zeros((0,))
            valid_names_new = []
        valid_names = valid_names_new

        np.save(embedding_path, embeddings)
        np.save(name_path, np.array(valid_names))
        print(f"Saved updated embeddings shape: {embeddings.shape}")
        need_compute = False

if need_compute:
    print("Computing all embeddings ...")
    embeddings = compute_embeddings(images, preprocess, model, device)
    np.save(embedding_path, embeddings)
    np.save(name_path, np.array(valid_names))
    print(f"Saved embeddings shape: {embeddings.shape}")


# %% Select subset of images by name for visualization
# Example: filter by substrings or provide an explicit list
# To filter, adjust filter_image_names or filter_substrings as needed.

# Option 1: Filter by explicit list
filter_image_names = None # ['circle_cross_red.png', 'circle_cross_blue.png', 'circle_cross_green.png'] 
# Option 2: Filter by substring matching
filter_substrings = []#["none_","man_","grandma_","boy_"]  

if filter_image_names is not None:
    selected_mask = [name in filter_image_names for name in valid_names]

elif filter_substrings:
    def match_substring_in_name(name, substr):
        # Exact match
        if name == substr:
            return True
        # '_substr_' - match only at start or on word boundary
        if name.startswith(substr):
            # make sure not treating char before as underscore
            return True
        if f"_{substr}" in name:
            # Split by underscore, check for substr between underscores or after
            parts = name.split('_')
            for part in parts[1:]:
                if part.startswith(substr):
                    return True
        return False

    selected_mask = [
        any(
            # Must not be preceeded by '_'
            (
                name.find(substr) != -1 and 
                (
                    name.find(substr) == 0 or name[name.find(substr) - 1] != '_'
                )
            ) if substr != "" else False
            for substr in filter_substrings
        )
        for name in valid_names
    ]
else:
    selected_mask = [True] * len(valid_names)

selected_embeddings = embeddings[selected_mask]
selected_names = [name for i, name in enumerate(valid_names) if selected_mask[i]]

print(f"Selected {len(selected_names)}/{len(valid_names)} images for visualization.")



# %% default order
# --------- 1. Plot similarity matrix 
sorted_indices = sorted(range(len(selected_names)), key=lambda i: selected_names[i])
sorted_names = [selected_names[i] for i in sorted_indices]
sorted_embeddings = selected_embeddings[sorted_indices]

sim_matrix = cosine_similarity(sorted_embeddings)

plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_matrix, xticklabels=sorted_names, yticklabels=sorted_names,
    cmap="viridis", vmin=0.0, vmax=1.0, square=True, annot=False, cbar_kws={'label': 'Cosine similarity'}
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Image Embedding Cosine Similarity (Filtered, Alphabetical Order)")
plt.tight_layout()
plt.show()

# --------- 2. 
from umap.umap_ import UMAP

umap_model = UMAP(n_components=2, random_state=42)
embeddings_2d = umap_model.fit_transform(selected_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], alpha=0.7)
for i, name in enumerate(selected_names):
    plt.text(embeddings_2d[i,0], embeddings_2d[i,1], name, fontsize=8, alpha=0.8)
plt.title("Image embeddings - UMAP 2D projection (Filtered)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

# %%
# 将 name 中 '_' 连接的部分拆开，并倒装顺序，但不要倒装 .png 后缀
def reverse_name_keep_ext(name):
    # 如果有扩展名（.png），保留扩展名
    if name.lower().endswith('.png'):
        base = name[:-4]
        ext = name[-4:]
    else:
        base = name
        ext = ''
    parts = base.split('_')
    reversed_name = '_'.join(parts[::-1]) + ext
    return reversed_name

reversed_names = [reverse_name_keep_ext(name) for name in selected_names]

# 按新的 name 的字母表排序
sorted_indices_reversed = sorted(range(len(reversed_names)), key=lambda i: reversed_names[i])
sorted_names_reversed = [reversed_names[i] for i in sorted_indices_reversed]
sorted_embeddings_reversed = selected_embeddings[sorted_indices_reversed]

# 再次绘图（可以是 similarity matrix，也可以是 2D embedding）
# 这里以 similarity matrix 为例

sim_matrix_reversed = cosine_similarity(sorted_embeddings_reversed)

plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_matrix_reversed, 
    xticklabels=sorted_names_reversed, 
    yticklabels=sorted_names_reversed,
    cmap="viridis", vmin=0.0, vmax=1.0, square=True, annot=False, cbar_kws={'label': 'Cosine similarity'}
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Image Embedding Cosine Similarity (Reversed Names, Alphabetical Order)")
plt.tight_layout()
plt.show()

# 2D UMAP 按照倒装名称排序后再画一遍
umap_model_rev = UMAP(n_components=2, random_state=42)
embeddings_2d_reversed = umap_model_rev.fit_transform(sorted_embeddings_reversed)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d_reversed[:,0], embeddings_2d_reversed[:,1], alpha=0.7)
for i, name in enumerate(sorted_names_reversed):
    plt.text(embeddings_2d_reversed[i,0], embeddings_2d_reversed[i,1], name, fontsize=8, alpha=0.8)
plt.title("Image embeddings - UMAP 2D projection (Reversed Names Alphabetical)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

# %% Embedding arithmetic test

def embedding_arithmetic_test(
    src_names,         # list of image names
    src_embeddings,    # embedding array, shape (N, D)
    base_names,        # list of names for base images (e.g. ["A", "B"])
    op_names,          # list of names for operator images (e.g. ["C"])
    target_name,       # target image name (e.g. "result")
    coeffs=None,       # list of coeffs for base & operator images, same length as base_names+op_names
    verbose=True
):
    """
    Perform embedding arithmetic: (sum_i coeffs[i]*embedding[name[i]]) vs embedding[target_name]
    Returns the cosine similarity and (optional) the normed vectors.

    Example: embedding_arithmetic_test(selected_names, selected_embeddings, ["A","B"], [], "C", [1, 1])

    Args:
        src_names: list of all image names
        src_embeddings: corresponding 2D ndarray
        base_names: names whose embeddings are to be summed/combined (list)
        op_names: other names whose embeddings (e.g., subtract) in combination (can be [])
        target_name: the name to compare arithmetic result to
        coeffs: List of coefficients for each base/op_name (default: 1 for each base, -1 for each op)
        verbose: whether to print details
    """
    # Helper function
    def find_index(names_list, target):
        try:
            return names_list.index(target)
        except ValueError:
            if verbose:
                print(f"Name '{target}' not found in source names.")
            return None

    # Collect indices
    indices = [find_index(src_names, n) for n in base_names + op_names]
    target_idx = find_index(src_names, target_name)

    # If not all found, abort and warn
    if any(idx is None for idx in indices+[target_idx]):
        if verbose:
            print("Some names not found. Aborting embedding arithmetic.")
        return None

    # Assign default coeffs: +1 for base, -1 for op
    n_base = len(base_names)
    n_op = len(op_names)
    if coeffs is None:
        coeffs = [1.0]*n_base + [-1.0]*n_op
    assert len(coeffs) == n_base + n_op, "Length of coeffs must match sum of base_names and op_names"

    # Compute arithmetic embedding
    emb_result = sum(coeff * src_embeddings[idx] for coeff, idx in zip(coeffs, indices))
    emb_target = src_embeddings[target_idx]

    # Normalize
    def normalize(vec):
        return vec / (np.linalg.norm(vec) + 1e-8)
    emb_result_norm = normalize(emb_result)
    emb_target_norm = normalize(emb_target)

    # Cosine similarity
    sim = np.dot(emb_result_norm, emb_target_norm)

    if verbose:
        op_expr = " + ".join([f"{c}*{n}" for c, n in zip(coeffs, base_names+op_names)])
        print(f"Cosine similarity between ({op_expr}) and {target_name}: {sim:.4f}")

    return sim, emb_result_norm, emb_target_norm


# %%
shape_list = ["shape", 'circle', 'square', 'triangle', 'star', 'cross']
color_list = ["color", 'red', 'blue', 'green', 'yellow', 'purple']
texture_list = ["texture", 'slash', 'cross', 'dot', 'star', 'dash']

for text_list in [shape_list, texture_list, color_list]:
    # 第一步：提取embedding
    with torch.no_grad():
        if 'open_clip' in globals():
            text_tokens = tokenizer(text_list)
            text_tokens = text_tokens.to(device)
            text_features = model.encode_text(text_tokens)
        else:
            text_tokens = tokenizer(text_list).to(device)
            text_features = model.encode_text(text_tokens)

    # 第二步：求norm
    emb_norm = text_features.norm(dim=1).cpu().numpy()
    print(f"Embedding L2 norm for each text: {emb_norm}")


# %%
