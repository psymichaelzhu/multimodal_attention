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
target_dir = 'data/lab_stimuli_v4'
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

# %% --------- Compute or load CLIP embeddings ---------
embedding_dir = os.path.join(target_dir, 'embedding')
os.makedirs(embedding_dir, exist_ok=True)
embedding_path = os.path.join(embedding_dir, 'picture_embedding.npy')
name_path = os.path.join(embedding_dir, 'picture_name_list.npy')
overwrite = False

if os.path.exists(embedding_path) and os.path.exists(name_path) and not overwrite:
    print("Loading existing embeddings ...")
    embeddings = np.load(embedding_path)
    valid_names = list(np.load(name_path, allow_pickle=True))
    print(f"Loaded embeddings shape: {embeddings.shape}")
else:
    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch_imgs = images[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            batch_emb = model.encode_image(batch_tensors)
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0).numpy()

    # save
    np.save(embedding_path, embeddings)
    np.save(name_path, np.array(valid_names))
    print(f"Saved embeddings shape: {embeddings.shape}")


# 

# %% Select subset of images by name for visualization
# Example: filter by substrings or provide an explicit list
# To filter, adjust filter_image_names or filter_substrings as needed.

# Option 1: Filter by explicit list
filter_image_names = None  # e.g. ['cat1.jpg', 'dog2.png']
# Option 2: Filter by substring matching
filter_substrings = []  # e.g. ['cat', 'dog']

if filter_image_names is not None:
    selected_mask = [name in filter_image_names for name in valid_names]
elif filter_substrings:
    selected_mask = [
        any(substr in name for substr in filter_substrings)
        for name in valid_names
    ]
else:
    selected_mask = [True] * len(valid_names)

selected_embeddings = embeddings[selected_mask]
selected_names = [name for i, name in enumerate(valid_names) if selected_mask[i]]

print(f"Selected {len(selected_names)}/{len(valid_names)} images for visualization.")

# --------- 1. Plot similarity matrix ---------
sim_matrix = cosine_similarity(selected_embeddings)

plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_matrix, xticklabels=selected_names, yticklabels=selected_names,
    cmap="viridis", vmin=0.0, vmax=1.0, square=True, annot=False, cbar_kws={'label': 'Cosine similarity'}
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Image Embedding Cosine Similarity (Filtered)")
plt.tight_layout()
plt.show()

# --------- 2. 2D PCA visualization ---------
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(selected_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], alpha=0.7)
for i, name in enumerate(selected_names):
    plt.text(embeddings_2d[i,0], embeddings_2d[i,1], name, fontsize=8, alpha=0.8)
plt.title("Image embeddings - PCA 2D projection (Filtered)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# %%
