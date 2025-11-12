# %% objective
# test whether the norm length reflects the informativeness (more dimensions) of the embedding
# %% preparation
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# set working directory
os.chdir('/Users/rezek_zhu/multimodal_attention')  
print(os.getcwd())

# model info
model_source, model_name, pretrained_name =  ('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b')

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
if model_source == "openclip":
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
elif model_source == "clip":
    import clip
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize

# %% obtain embeddings
# helper function
def extract_embedding(objects, type="image",normalize=False):
    """
    Extract embedding using CLIP model based on input type

    input: 
        objects (image or text): a list of objects to extract embedding for. For image type, each element should be `numpy array` or `PIL image`. For text type, each element should be `string`.
        type (str): Type of objects, either "image" or "text"
    output: 
        embedding (tensor): Extracted features from CLIP model, which should be a tensor of shape (len(objects), embedding_dim).
    """
    if type == "image":
        image = [preprocess(obj if isinstance(obj, Image.Image) else Image.fromarray(obj)) for obj in objects] # list of tensors
        with torch.no_grad():
            features = model.encode_image(torch.stack(image).to(device))
    elif type == "text":
        text = tokenizer(objects).to(device)
        with torch.no_grad():
            features = model.encode_text(text)
    if normalize:
        features = features / features.norm(dim=-1, keepdim=True)
    return features
# %% load images from data/image_text_alignment/picture
base_dir = 'data/genAI_stimuli2'
image_dir = os.path.join(base_dir, 'picture')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]# and ('woman' in f.lower() or 'boy' in f.lower())]

# load images
images = []
image_names = []
descriptions = []
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    try:
        image = Image.open(img_path).convert('RGB')
        # remove file extension to get image name
        image_name = os.path.splitext(img_file)[0]
    
        images.append(image)
        image_names.append(image_name)
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

#%% extract embeddings
# extract image embeddings
image_embeddings = extract_embedding(images, type="image")
# normalize embeddings for cosine similarity
image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)


#%% plot correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns
def plot_similarity_heatmap(embeddings1, embeddings2, labels1, labels2, title, xlabel, ylabel, figsize=(10, 8), scale_to_01=False):
    """
    Plot similarity heatmap between two sets of embeddings
    
    Args:
        embeddings1: First set of normalized embeddings (tensor)
        embeddings2: Second set of normalized embeddings (tensor)
        labels1: Labels for rows (list)
        labels2: Labels for columns (list)
        title: Title of the heatmap (str)
        xlabel: Label for x-axis (str)
        ylabel: Label for y-axis (str)
        figsize: Figure size (tuple)
        scale_to_01: Whether to scale similarity scores to 0-1 range (bool)
    """
    similarity_matrix = (embeddings1 @ embeddings2.T).cpu().numpy()
    
    if scale_to_01:
        # Scale similarity matrix to 0-1 range
        min_val = similarity_matrix.min()
        max_val = similarity_matrix.max()
        if max_val > min_val:
            similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis', 
                xticklabels=labels2, 
                yticklabels=labels1,
                cbar_kws={'label': 'Cosine Similarity'},
                vmin=0, vmax=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Return only the lower triangle of the similarity matrix
    return np.tril(similarity_matrix)

# plot within-image similarity heatmap
image_image_similarity = plot_similarity_heatmap(image_embeddings_norm, image_embeddings_norm,
                        image_names, image_names,
                        'Within-Image Embedding Similarity Heatmap',
                        'Images', 'Images')

# compute the correlation between the norms of image and text embeddings
image_norms = image_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
print(image_norms)
# %%
# extract text embeddings for image names
text_embeddings = extract_embedding(image_names, type="text")

# normalize text embeddings
text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# plot within-text similarity heatmap
text_text_similarity = plot_similarity_heatmap(text_embeddings_norm, text_embeddings_norm,
                        image_names, image_names,
                        'Within-Text Embedding Similarity Heatmap',
                        'Text Descriptions', 'Text Descriptions',
                        scale_to_01=False)

# plot text-image similarity heatmap
text_image_similarity = plot_similarity_heatmap(text_embeddings_norm, image_embeddings_norm,
                        image_names, image_names,
                        'Text-Image Embedding Alignment Heatmap',
                        'Images', 'Text Descriptions',
                        scale_to_01=False)

text_norms = text_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
print(text_norms)
# %%
# Create scatter plots for similarity and norms
import matplotlib.pyplot as plt

# Extract lower triangle values (excluding diagonal) for similarity
n = len(image_names)
image_image_sim_values = []
text_text_sim_values = []
text_image_sim_values = []

for i in range(n):
    for j in range(i):
        image_image_sim_values.append(image_image_similarity[i, j])
        text_text_sim_values.append(text_text_similarity[i, j])
        text_image_sim_values.append(text_image_similarity[i, j])

# First plot: Similarity scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Image similarity
ax1.scatter(range(len(image_image_sim_values)), image_image_sim_values, alpha=0.6, s=50)
ax1.set_xlabel('Pair Index')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Within-Image Similarity')
ax1.grid(True, alpha=0.3)

# Right: Text similarity
ax2.scatter(range(len(text_text_sim_values)), text_text_sim_values, alpha=0.6, s=50, color='orange')
ax2.set_xlabel('Pair Index')
ax2.set_ylabel('Cosine Similarity')
ax2.set_title('Within-Text Similarity')
ax2.grid(True, alpha=0.3)

# Set same y-axis limits for both plots
sim_ylim = (min(min(image_image_sim_values), min(text_text_sim_values)), 
            max(max(image_image_sim_values), max(text_text_sim_values)))
ax1.set_ylim(sim_ylim)
ax2.set_ylim(sim_ylim)

plt.tight_layout()
plt.show()

# Second plot: Norm scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Image norms
ax1.scatter(range(len(image_norms)), image_norms.flatten(), alpha=0.6, s=50)
ax1.set_xlabel('Image Index')
ax1.set_ylabel('Embedding Norm')
ax1.set_title('Image Embedding Norms')
ax1.grid(True, alpha=0.3)

# Right: Text norms
ax2.scatter(range(len(text_norms)), text_norms.flatten(), alpha=0.6, s=50, color='orange')
ax2.set_xlabel('Text Index')
ax2.set_ylabel('Embedding Norm')
ax2.set_title('Text Embedding Norms')
ax2.grid(True, alpha=0.3)

# Set same y-axis limits for both plots
norm_ylim = (min(image_norms.min(), text_norms.min()), 
             max(image_norms.max(), text_norms.max()))
ax1.set_ylim(norm_ylim)
ax2.set_ylim(norm_ylim)

plt.tight_layout()
plt.show()

# %%
def analyze_increasing_information(description_dict, add_prefix=False):
    """
    Analyze how embedding norm changes with increasing information levels
    
    Args:
        description_dict: Dictionary with keys as levels (e.g., "L1", "L2", ...) and values as descriptions
        add_prefix: Whether to add a prefix to descriptions (default: False)
    
    Returns:
        norms: Array of norm values for each level
    """
    # Extract text embeddings for increasing information description
    descriptions = [description for description in description_dict.values()]
    if add_prefix:
        descriptions = [f"A photo of {desc}" for desc in descriptions]
    
    embeddings = extract_embedding(descriptions, type="text")
    
    # Calculate norms
    norms = embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
    print(norms)
    
    # Plot norm length vs level index
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(norms) + 1), norms.flatten(), marker='o', linewidth=2, markersize=8)
    plt.xlabel('Level Index')
    plt.ylabel('Norm Length')
    plt.title('Embedding Norm Length vs Information Level')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(norms) + 1))
    plt.tight_layout()
    plt.show()
    
    return norms


# Define increasing information descriptions
increasing_information_description_dict = {
    "man": {
        "L1": "a man",
        "L2": "a tall man",
        "L3": "a tall bearded man",
        "L4": "a tall bearded young man",
        "L5": "a tall bearded young slim man",
        "L6": "a tall bearded young slim smiling man",
        "L7": "a tall bearded young slim smiling Asian man",
        "L8": "a tall bearded young slim smiling Asian casually-dressed man",
        "L9": "a tall bearded young slim smiling Asian casually-dressed red-shirted man",
        "L10": "a tall bearded young slim smiling Asian casually-dressed red-shirted friendly-looking man"
    },
    "car": {
        "L1": "A car",
        "L2": "A red car",
        "L3": "A red compact car",
        "L4": "A red compact modern car",
        "L5": "A red compact modern electric car",
        "L6": "A red compact modern electric aerodynamic car",
        "L7": "A red compact modern electric aerodynamic sleek car",
        "L8": "A red compact modern electric aerodynamic sleek glossy car",
        "L9": "A red compact modern electric aerodynamic sleek glossy two-door car",
        "L10": "A red compact modern electric aerodynamic sleek glossy two-door lightweight car"
    },
    "dog": {
        "L1": "A dog",
        "L2": "A brown dog",
        "L3": "A brown fluffy dog",
        "L4": "A brown fluffy energetic dog",
        "L5": "A brown fluffy energetic medium-sized dog",
        "L6": "A brown fluffy energetic medium-sized friendly dog",
        "L7": "A brown fluffy energetic medium-sized friendly playful dog",
        "L8": "A brown fluffy energetic medium-sized friendly playful outdoor dog",
        "L9": "A brown fluffy energetic medium-sized friendly playful outdoor happy dog",
        "L10": "A brown fluffy energetic medium-sized friendly playful outdoor happy tail-wagging dog"
        },
    "bird": {
        "L1": "A bird",
        "L2": "A small bird",
        "L3": "A small yellow bird",
        "L4": "A small yellow bright-feathered bird",
        "L5": "A small yellow bright-feathered singing bird",
        "L6": "A small yellow bright-feathered singing delicate bird",
        "L7": "A small yellow bright-feathered singing delicate high-perched bird",
        "L8": "A small yellow bright-feathered singing delicate high-perched cheerful bird",
        "L9": "A small yellow bright-feathered singing delicate high-perched cheerful forest bird",
        "L10": "A small yellow bright-feathered singing delicate high-perched cheerful forest wild bird"
        },
    "chair": {
        "L1": "A chair",
        "L2": "A wooden chair",
        "L3": "A wooden sturdy chair",
        "L4": "A wooden sturdy simple chair",
        "L5": "A wooden sturdy simple light-brown chair",
        "L6": "A wooden sturdy simple light-brown straight-backed chair",
        "L7": "A wooden sturdy simple light-brown straight-backed polished chair",
        "L8": "A wooden sturdy simple light-brown straight-backed polished minimalist chair",
        "L9": "A wooden sturdy simple light-brown straight-backed polished minimalist dining chair",
        "L10": "A wooden sturdy simple light-brown straight-backed polished minimalist dining smooth-edged chair"
        },
    "cup": {
        "L1": "A cup",
        "L2": "A ceramic cup",
        "L3": "A ceramic white cup",
        "L4": "A ceramic white small cup",
        "L5": "A ceramic white small round-edged cup",
        "L6": "A ceramic white small round-edged glossy cup",
        "L7": "A ceramic white small round-edged glossy heat-resistant cup",
        "L8": "A ceramic white small round-edged glossy heat-resistant smoothly-shaped cup",
        "L9": "A ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist cup",
        "L10": "A ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist lightweight cup"
        },
    "kitchen": {
        "L1": "A kitchen",
        "L2": "A modern kitchen",
        "L3": "A modern bright kitchen",
        "L4": "A modern bright clean kitchen",
        "L5": "A modern bright clean spacious kitchen",
        "L6": "A modern bright clean spacious organized kitchen",
        "L7": "A modern bright clean spacious organized white-colored kitchen",
        "L8": "A modern bright clean spacious organized white-colored stainless-steel kitchen",
        "L9": "A modern bright clean spacious organized white-colored stainless-steel well-lit kitchen",
        "L10": "A modern bright clean spacious organized white-colored stainless-steel well-lit minimalist kitchen"
    },
    "forest": {
        "L1": "A forest",
        "L2": "A dense forest",
        "L3": "A dense green forest",
        "L4": "A dense green misty forest",
        "L5": "A dense green misty tranquil forest",
        "L6": "A dense green misty tranquil tall-tree forest",
        "L7": "A dense green misty tranquil tall-tree sunlit forest",
        "L8": "A dense green misty tranquil tall-tree sunlit moss-covered forest",
        "L9": "A dense green misty tranquil tall-tree sunlit moss-covered nature-rich forest",
        "L10": "A dense green misty tranquil tall-tree sunlit moss-covered nature-rich atmospheric forest"
    },
}

#%%
# Analyze man descriptions
man_norms = analyze_increasing_information(increasing_information_description_dict["man"], add_prefix=True)

# %%
# Analyze car descriptions
car_norms = analyze_increasing_information(increasing_information_description_dict["car"], add_prefix=True)

# %%
for key in increasing_information_description_dict.keys():
    print(key)
    norms = analyze_increasing_information(increasing_information_description_dict[key], add_prefix=True)
    print("--------------------------------")
# %%

# how to think about exceptions?


# %% 
# Simulation function for random description sampling
def simulate_random_descriptions(description_list, target_list, n_sim_per_L):
    """
    Generate random combinations of descriptions with increasing levels
    
    Args:
        description_list: List of description words (e.g., ['tall', 'bearded', 'young', ...])
        target_list: List of target words (e.g., ['one', 'man'])
        n_sim_per_L: Number of simulations per level
    
    Returns:
        text_list: List of all generated text descriptions
        level_indices: List of level indices corresponding to each text
    """
    text_list = []
    level_indices = []
    
    # For each level (1 to len(description_list))
    for level_idx in range(1, len(description_list) + 1):
        # Repeat n_sim_per_L times for this level
        for _ in range(n_sim_per_L):
            # Sample level_idx descriptions without replacement
            sampled_descriptions = np.random.choice(description_list, size=level_idx, replace=False)
            # Join with commas and add target
            text = target_list[0] + " " + ", ".join(sampled_descriptions) + " " + target_list[1]
            text_list.append(text)
            level_indices.append(level_idx)
    
    return text_list, level_indices


def plot_random_descriptions(description, target, n_sim_per_L):
    # Generate random descriptions
    text_list, level_indices = simulate_random_descriptions(description, target, n_sim_per_L)

    # Extract embeddings
    text_embeddings = extract_embedding(text_list, type="text", normalize=False)
    text_norms = text_embeddings.norm(dim=-1).cpu().numpy()

    # Plot norm vs. level with scatter points
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot scatter points for each simulation
    for level in range(1, len(description) + 1):
        # Get indices for this level
        level_mask = np.array(level_indices) == level
        level_norms = text_norms[level_mask]
        # Add small jitter for visibility
        x_positions = np.ones(len(level_norms)) * level + np.random.normal(0, 0.05, len(level_norms))
        plt.scatter(x_positions, level_norms, alpha=0.6, s=50)

    # Plot mean line
    level_means = [text_norms[np.array(level_indices) == level].mean() 
                for level in range(1, len(description) + 1)]
    plt.plot(range(1, len(description) + 1), level_means, 'r-', linewidth=2, label='Mean')

    plt.xlabel('Number of Descriptions (Level)', fontsize=12)
    plt.ylabel('Embedding Norm', fontsize=12)
    plt.title(f'Embedding Norm vs. Description Level (n_sim={n_sim_per_L})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%

# Example usage
description = ['tall', 'bearded', 'young', 'slim', 'smiling', 'Asian', 
               'casually-dressed', 'red-shirted', 'friendly-looking']
target = ['one', 'man']
n_sim_per_L = 100
plot_random_descriptions(description, target, n_sim_per_L)
# variation due to description content
# %%
