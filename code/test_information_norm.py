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

# %% main
# %% helper functions
# %% helper functions
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


# Simulation function for random description sampling
def simulate_random_descriptions(description, n_sim_per_L):
    """
    Generate random combinations of descriptions with increasing levels
    
    Args:
        description: Full description string (e.g., 'one tall bearded young slim man')
        n_sim_per_L: Number of simulations per level
    
    Returns:
        text_list: List of all generated text descriptions
        level_indices: List of level indices corresponding to each text
    """
    # Split description into words
    words = description.split()
    
    # Extract first and last words as target_list
    target_list = [words[0], words[-1]]
    
    # Extract middle words as description_list
    description_list = words[1:-1]
    
    text_list = []
    level_indices = []
    description_sampled_list = []
    
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
            description_sampled_list.append(sampled_descriptions)
    return text_list, level_indices, description_list, description_sampled_list

def plot_random_descriptions(description, n_sim_per_L, add_prefix=False,dot_size=100):
    # Generate random descriptions
    text_list, level_indices, description_list, description_sampled_list = simulate_random_descriptions(description, n_sim_per_L)

    description_list_embedding = extract_embedding(description_list, type="text", normalize=False)
    description_list_norm = description_list_embedding.norm(dim=-1).cpu().numpy()
    
    # Create a mapping from description to norm
    description_to_norm = {desc: norm for desc, norm in zip(description_list, description_list_norm)}
    
    # For each sampled set, compute the mean norm
    description_sampled_norm = []
    for sampled_set in description_sampled_list:
        norms = [description_to_norm[desc] for desc in sampled_set]
        description_sampled_norm.append(np.mean(norms))

    if add_prefix:
        text_list = [f"A photo of {text}" for text in text_list]
    # Extract embeddings
    text_embeddings = extract_embedding(text_list, type="text", normalize=False)
    text_norms = text_embeddings.norm(dim=-1).cpu().numpy()
    
    
    # Plot norm vs. level with scatter points
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sample points for plotting if n_sim_per_L > 100
    if n_sim_per_L > 100:
        # Get unique levels
        unique_levels = sorted(set(level_indices))
        sampled_indices = []
        
        for level in unique_levels:
            # Get all indices for this level
            level_mask = np.array(level_indices) == level
            level_idx_positions = np.where(level_mask)[0]
            
            # Sample 100 points from this level
            sampled_idx = np.random.choice(level_idx_positions, size=min(100, len(level_idx_positions)), replace=False)
            sampled_indices.extend(sampled_idx)
        
        # Use sampled indices for plotting
        plot_level_indices = [level_indices[i] for i in sampled_indices]
        plot_text_norms = text_norms[sampled_indices]
        plot_description_sampled_norm = [description_sampled_norm[i] for i in sampled_indices]
    else:
        plot_level_indices = level_indices
        plot_text_norms = text_norms
        plot_description_sampled_norm = description_sampled_norm
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(
        plot_level_indices,
        plot_text_norms,
        alpha=0.6,
        s=dot_size,
        c=plot_description_sampled_norm,
        cmap='viridis'
    )
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Mean Description Norm')
    
    
    # Add polynomial regression line (use all data for regression)
    sns.regplot(
        x=level_indices,
        y=text_norms,
        order=2,
        scatter=False,
        line_kws={'color': 'r', 'alpha': 0.4},
        ax=ax
    )
    
    ax.set_xlabel('Number of Descriptions (Level)', fontsize=12)
    ax.set_ylabel('Embedding Norm', fontsize=12)
    ax.set_title(f'{description} (n_sim={n_sim_per_L})', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    if True:
        # Plot correlation between sampled_norm and text_norm
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Get unique levels and assign colors
        unique_levels = sorted(set(level_indices))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_levels)))
        
        # Plot each level with a different color
        for i, level in enumerate(unique_levels):
            # Get indices for this level
            level_mask = np.array(level_indices) == level
            level_sampled_norm = np.array(description_sampled_norm)[level_mask]
            level_text_norms = text_norms[level_mask]
            
            # Sample points for plotting if n_sim_per_L > 100
            if n_sim_per_L > 100:
                sample_size = min(100, len(level_text_norms))
                sample_idx = np.random.choice(len(level_text_norms), size=sample_size, replace=False)
                level_sampled_norm = level_sampled_norm[sample_idx]
                level_text_norms = level_text_norms[sample_idx]
            
            # Scatter plot for this level
            ax2.scatter(level_sampled_norm, level_text_norms, 
                    alpha=0.6, s=dot_size, color=colors[i], 
                    label=f'Level {level}')
            
            # Add overall regression line (use all data for this level)
            level_mask_full = np.array(level_indices) == level
            level_sampled_norm_full = np.array(description_sampled_norm)[level_mask_full]
            level_text_norms_full = text_norms[level_mask_full]
            
            sns.regplot(
                x=level_sampled_norm_full,
                y=level_text_norms_full,
                scatter=False,
                line_kws={'color': colors[i], 'alpha': 0.6},
                ax=ax2
            )
        
        ax2.set_xlabel('Mean Norm of Sampled Descriptions', fontsize=12)
        ax2.set_ylabel('Text Embedding Norm', fontsize=12)
        ax2.set_title(f'Correlation: Sampled Description Norm vs Text Norm\n{description}', fontsize=14)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# %% Define increasing information descriptions
increasing_information_description_dict = {
    "man": {
        "L1": "one man",
        "L2": "one tall man",
        "L3": "one tall bearded man",
        "L4": "one tall bearded young man",
        "L5": "one tall bearded young slim man",
        "L6": "one tall bearded young slim smiling man",
        "L7": "one tall bearded young slim smiling Asian man",
        "L8": "one tall bearded young slim smiling Asian casually-dressed man",
        "L9": "one tall bearded young slim smiling Asian casually-dressed red-shirted man",
        "L10": "one tall bearded young slim smiling Asian casually-dressed red-shirted friendly-looking man"
    },
    "car": {
        "L1": "one car",
        "L2": "one red car",
        "L3": "one red compact car",
        "L4": "one red compact modern car",
        "L5": "one red compact modern electric car",
        "L6": "one red compact modern electric aerodynamic car",
        "L7": "one red compact modern electric aerodynamic sleek car",
        "L8": "one red compact modern electric aerodynamic sleek glossy car",
        "L9": "one red compact modern electric aerodynamic sleek glossy two-door car",
        "L10": "one red compact modern electric aerodynamic sleek glossy two-door lightweight car"
    },
    "dog": {
        "L1": "one dog",
        "L2": "one brown dog",
        "L3": "one brown fluffy dog",
        "L4": "one brown fluffy energetic dog",
        "L5": "one brown fluffy energetic medium-sized dog",
        "L6": "one brown fluffy energetic medium-sized friendly dog",
        "L7": "one brown fluffy energetic medium-sized friendly playful dog",
        "L8": "one brown fluffy energetic medium-sized friendly playful outdoor dog",
        "L9": "one brown fluffy energetic medium-sized friendly playful outdoor happy dog",
        "L10": "one brown fluffy energetic medium-sized friendly playful outdoor happy tail-wagging dog"
        },
    "bird": {
        "L1": "one bird",
        "L2": "one small bird",
        "L3": "one small yellow bird",
        "L4": "one small yellow bright-feathered bird",
        "L5": "one small yellow bright-feathered singing bird",
        "L6": "one small yellow bright-feathered singing delicate bird",
        "L7": "one small yellow bright-feathered singing delicate high-perched bird",
        "L8": "one small yellow bright-feathered singing delicate high-perched cheerful bird",
        "L9": "one small yellow bright-feathered singing delicate high-perched cheerful forest bird",
        "L10": "one small yellow bright-feathered singing delicate high-perched cheerful forest wild bird"
        },
    "chair": {
        "L1": "one chair",
        "L2": "one wooden chair",
        "L3": "one wooden sturdy chair",
        "L4": "one wooden sturdy simple chair",
        "L5": "one wooden sturdy simple light-brown chair",
        "L6": "one wooden sturdy simple light-brown straight-backed chair",
        "L7": "one wooden sturdy simple light-brown straight-backed polished chair",
        "L8": "one wooden sturdy simple light-brown straight-backed polished minimalist chair",
        "L9": "one wooden sturdy simple light-brown straight-backed polished minimalist dining chair",
        "L10": "one wooden sturdy simple light-brown straight-backed polished minimalist dining smooth-edged chair"
        },
    "cup": {
        "L1": "one cup",
        "L2": "one ceramic cup",
        "L3": "one ceramic white cup",
        "L4": "one ceramic white small cup",
        "L5": "one ceramic white small round-edged cup",
        "L6": "one ceramic white small round-edged glossy cup",
        "L7": "one ceramic white small round-edged glossy heat-resistant cup",
        "L8": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped cup",
        "L9": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist cup",
        "L10": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist lightweight cup"
        },
    "kitchen": {
        "L1": "one kitchen",
        "L2": "one modern kitchen",
        "L3": "one modern bright kitchen",
        "L4": "one modern bright clean kitchen",
        "L5": "one modern bright clean spacious kitchen",
        "L6": "one modern bright clean spacious organized kitchen",
        "L7": "one modern bright clean spacious organized white-colored kitchen",
        "L8": "one modern bright clean spacious organized white-colored stainless-steel kitchen",
        "L9": "one modern bright clean spacious organized white-colored stainless-steel well-lit kitchen",
        "L10": "one modern bright clean spacious organized white-colored stainless-steel well-lit minimalist kitchen"
    },
    "forest": {
        "L1": "one forest",
        "L2": "one dense forest",
        "L3": "one dense green forest",
        "L4": "one dense green misty forest",
        "L5": "one dense green misty tranquil forest",
        "L6": "one dense green misty tranquil tall-tree forest",
        "L7": "one dense green misty tranquil tall-tree sunlit forest",
        "L8": "one dense green misty tranquil tall-tree sunlit moss-covered forest",
        "L9": "one dense green misty tranquil tall-tree sunlit moss-covered nature-rich forest",
        "L10": "one dense green misty tranquil tall-tree sunlit moss-covered nature-rich atmospheric forest"
    },
}

# %% law 1: more description, shorter norm
for key in increasing_information_description_dict.keys():
    plot_random_descriptions(increasing_information_description_dict[key]["L10"], n_sim_per_L=200, add_prefix=True)
# %%
increasing_information_description_dict2 = {
    "woman": {
        "L1": "one woman",
        "L2": "one young woman",
        "L3": "one young smiling woman",
        "L4": "one young smiling blonde woman",
        "L5": "one young smiling blonde tall woman",
        "L6": "one young smiling blonde tall casually-dressed woman",
        "L7": "one young smiling blonde tall casually-dressed confident woman",
        "L8": "one young smiling blonde tall casually-dressed confident stylish woman",
        "L9": "one young smiling blonde tall casually-dressed confident stylish city-strolling woman",
        "L10": "one young smiling blonde tall casually-dressed confident stylish city-strolling friendly woman"
    },
    "cat": {
        "L1": "one cat",
        "L2": "one white cat",
        "L3": "one white fluffy cat",
        "L4": "one white fluffy calm cat",
        "L5": "one white fluffy calm small cat",
        "L6": "one white fluffy calm small sleeping cat",
        "L7": "one white fluffy calm small sleeping window-side cat",
        "L8": "one white fluffy calm small sleeping window-side domestic cat",
        "L9": "one white fluffy calm small sleeping window-side domestic content cat",
        "L10": "one white fluffy calm small sleeping window-side domestic content purring cat"
    },
    "tree": {
        "L1": "one tree",
        "L2": "one tall tree",
        "L3": "one tall green tree",
        "L4": "one tall green leafy tree",
        "L5": "one tall green leafy thick-trunked tree",
        "L6": "one tall green leafy thick-trunked sunlit tree",
        "L7": "one tall green leafy thick-trunked sunlit solitary tree",
        "L8": "one tall green leafy thick-trunked sunlit solitary hillside tree",
        "L9": "one tall green leafy thick-trunked sunlit solitary hillside wind-blown tree",
        "L10": "one tall green leafy thick-trunked sunlit solitary hillside wind-blown majestic tree"
    },
    "city": {
        "L1": "one city",
        "L2": "one large city",
        "L3": "one large modern city",
        "L4": "one large modern bustling city",
        "L5": "one large modern bustling coastal city",
        "L6": "one large modern bustling coastal high-rise city",
        "L7": "one large modern bustling coastal high-rise illuminated city",
        "L8": "one large modern bustling coastal high-rise illuminated night city",
        "L9": "one large modern bustling coastal high-rise illuminated night cosmopolitan city",
        "L10": "one large modern bustling coastal high-rise illuminated night cosmopolitan lively city"
    },
    "house": {
        "L1": "one house",
        "L2": "one small house",
        "L3": "one small wooden house",
        "L4": "one small wooden cozy house",
        "L5": "one small wooden cozy countryside house",
        "L6": "one small wooden cozy countryside light-blue house",
        "L7": "one small wooden cozy countryside light-blue porch-fronted house",
        "L8": "one small wooden cozy countryside light-blue porch-fronted well-kept house",
        "L9": "one small wooden cozy countryside light-blue porch-fronted well-kept flower-surrounded house",
        "L10": "one small wooden cozy countryside light-blue porch-fronted well-kept flower-surrounded charming house"
    },
    "book": {
        "L1": "one book",
        "L2": "one old book",
        "L3": "one old leather-bound book",
        "L4": "one old leather-bound heavy book",
        "L5": "one old leather-bound heavy dusty book",
        "L6": "one old leather-bound heavy dusty ancient book",
        "L7": "one old leather-bound heavy dusty ancient hand-written book",
        "L8": "one old leather-bound heavy dusty ancient hand-written ornate book",
        "L9": "one old leather-bound heavy dusty ancient hand-written ornate mysterious book",
        "L10": "one old leather-bound heavy dusty ancient hand-written ornate mysterious valuable book"
    },
    "mountain": {
        "L1": "one mountain",
        "L2": "one high mountain",
        "L3": "one high rocky mountain",
        "L4": "one high rocky snow-capped mountain",
        "L5": "one high rocky snow-capped distant mountain",
        "L6": "one high rocky snow-capped distant majestic mountain",
        "L7": "one high rocky snow-capped distant majestic isolated mountain",
        "L8": "one high rocky snow-capped distant majestic isolated cloud-wrapped mountain",
        "L9": "one high rocky snow-capped distant majestic isolated cloud-wrapped silent mountain",
        "L10": "one high rocky snow-capped distant majestic isolated cloud-wrapped silent breathtaking mountain"
    },
    "table": {
        "L1": "one table",
        "L2": "one wooden table",
        "L3": "one wooden round table",
        "L4": "one wooden round polished table",
        "L5": "one wooden round polished dining table",
        "L6": "one wooden round polished dining sturdy table",
        "L7": "one wooden round polished dining sturdy minimalist table",
        "L8": "one wooden round polished dining sturdy minimalist light-brown table",
        "L9": "one wooden round polished dining sturdy minimalist light-brown clean table",
        "L10": "one wooden round polished dining sturdy minimalist light-brown clean elegant table"
    },
    "river": {
        "L1": "one river",
        "L2": "one calm river",
        "L3": "one calm clear river",
        "L4": "one calm clear flowing river",
        "L5": "one calm clear flowing narrow river",
        "L6": "one calm clear flowing narrow forest-side river",
        "L7": "one calm clear flowing narrow forest-side reflective river",
        "L8": "one calm clear flowing narrow forest-side reflective sunlit river",
        "L9": "one calm clear flowing narrow forest-side reflective sunlit tranquil river",
        "L10": "one calm clear flowing narrow forest-side reflective sunlit tranquil winding river"
    },
    "desk": {
        "L1": "one desk",
        "L2": "one wooden desk",
        "L3": "one wooden tidy desk",
        "L4": "one wooden tidy brown desk",
        "L5": "one wooden tidy brown minimalist desk",
        "L6": "one wooden tidy brown minimalist organized desk",
        "L7": "one wooden tidy brown minimalist organized office desk",
        "L8": "one wooden tidy brown minimalist organized office lamp-lit desk",
        "L9": "one wooden tidy brown minimalist organized office lamp-lit laptop-equipped desk",
        "L10": "one wooden tidy brown minimalist organized office lamp-lit laptop-equipped work-ready desk"
    }
}
for key in increasing_information_description_dict2.keys():
    plot_random_descriptions(increasing_information_description_dict2[key]["L10"], n_sim_per_L=200, add_prefix=True)

# %%
diverse_dimension_description_dict3 = {
    "man": {
        "L1": "one man",
        "L2": "one tall man",                      # physical height
        "L3": "one tall smiling man",              # emotion/expression
        "L4": "one tall smiling tired man",        # state/fatigue
        "L5": "one tall smiling tired office man", # occupation/context
        "L6": "one tall smiling tired office walking man", # action
        "L7": "one tall smiling tired office walking polite man", # social behavior
        "L8": "one tall smiling tired office walking polite bearded man", # appearance detail
        "L9": "one tall smiling tired office walking polite bearded thoughtful man", # mental state
        "L10": "one tall smiling tired office walking polite bearded thoughtful urban man" # environmental context
    },
    "woman": {
        "L1": "one woman",
        "L2": "one elegant woman",                 # style
        "L3": "one elegant cheerful woman",        # emotion
        "L4": "one elegant cheerful determined woman", # personality
        "L5": "one elegant cheerful determined artistic woman", # ability/interest
        "L6": "one elegant cheerful determined artistic traveling woman", # action
        "L7": "one elegant cheerful determined artistic traveling modern woman", # temporal/social context
        "L8": "one elegant cheerful determined artistic traveling modern red-dressed woman", # color
        "L9": "one elegant cheerful determined artistic traveling modern red-dressed confident woman", # trait
        "L10": "one elegant cheerful determined artistic traveling modern red-dressed confident metropolitan woman" # setting
    },
    "dog": {
        "L1": "one dog",
        "L2": "one brown dog",                     # color
        "L3": "one brown playful dog",             # temperament
        "L4": "one brown playful obedient dog",    # behavior trait
        "L5": "one brown playful obedient wet dog", # texture/state
        "L6": "one brown playful obedient wet running dog", # motion
        "L7": "one brown playful obedient wet running city dog", # location
        "L8": "one brown playful obedient wet running city park dog", # finer context
        "L9": "one brown playful obedient wet running city park curious dog", # cognition
        "L10": "one brown playful obedient wet running city park curious loyal dog" # moral trait
    },
    "car": {
        "L1": "one car",
        "L2": "one red car",                       # color
        "L3": "one red fast car",                  # speed/performance
        "L4": "one red fast quiet car",            # sound property
        "L5": "one red fast quiet electric car",   # mechanism
        "L6": "one red fast quiet electric compact car", # size
        "L7": "one red fast quiet electric compact city car", # usage context
        "L8": "one red fast quiet electric compact city rented car", # ownership
        "L9": "one red fast quiet electric compact city rented self-driving car", # function
        "L10": "one red fast quiet electric compact city rented self-driving futuristic car" # concept/futurity
    },
    "room": {
        "L1": "one room",
        "L2": "one bright room",                   # lighting
        "L3": "one bright clean room",             # cleanliness
        "L4": "one bright clean quiet room",       # sound
        "L5": "one bright clean quiet small room", # size
        "L6": "one bright clean quiet small modern room", # style
        "L7": "one bright clean quiet small modern cozy room", # comfort
        "L8": "one bright clean quiet small modern cozy windowed room", # spatial structure
        "L9": "one bright clean quiet small modern cozy windowed furnished room", # completeness
        "L10": "one bright clean quiet small modern cozy windowed furnished minimalist room" # design philosophy
    },
    "tree": {
        "L1": "one tree",
        "L2": "one tall tree",                     # size
        "L3": "one tall green tree",               # color
        "L4": "one tall green leafy tree",         # texture
        "L5": "one tall green leafy windy tree",   # dynamic condition
        "L6": "one tall green leafy windy riverside tree", # spatial context
        "L7": "one tall green leafy windy riverside ancient tree", # temporal
        "L8": "one tall green leafy windy riverside ancient twisted tree", # shape
        "L9": "one tall green leafy windy riverside ancient twisted sacred tree", # cultural meaning
        "L10": "one tall green leafy windy riverside ancient twisted sacred solitary tree" # social isolation / emotion
    },
    "street": {
        "L1": "one street",
        "L2": "one narrow street",                 # width
        "L3": "one narrow cobblestone street",     # material
        "L4": "one narrow cobblestone busy street", # activity
        "L5": "one narrow cobblestone busy market street", # function
        "L6": "one narrow cobblestone busy market colorful street", # visual
        "L7": "one narrow cobblestone busy market colorful European street", # geographic
        "L8": "one narrow cobblestone busy market colorful European festive street", # atmosphere
        "L9": "one narrow cobblestone busy market colorful European festive evening street", # time of day
        "L10": "one narrow cobblestone busy market colorful European festive evening rain-reflective street" # lighting/weather
    },
    "book": {
        "L1": "one book",
        "L2": "one old book",                      # age
        "L3": "one old heavy book",                # weight
        "L4": "one old heavy illustrated book",    # content
        "L5": "one old heavy illustrated historical book", # subject
        "L6": "one old heavy illustrated historical leather-bound book", # material
        "L7": "one old heavy illustrated historical leather-bound handwritten book", # authenticity
        "L8": "one old heavy illustrated historical leather-bound handwritten religious book", # theme
        "L9": "one old heavy illustrated historical leather-bound handwritten religious rare book", # rarity
        "L10": "one old heavy illustrated historical leather-bound handwritten religious rare valuable book" # value
    },
    "bird": {
        "L1": "one bird",
        "L2": "one small bird",                    # size
        "L3": "one small colorful bird",           # color
        "L4": "one small colorful singing bird",   # sound
        "L5": "one small colorful singing flying bird", # action
        "L6": "one small colorful singing flying tropical bird", # habitat
        "L7": "one small colorful singing flying tropical curious bird", # cognition
        "L8": "one small colorful singing flying tropical curious rare bird", # rarity
        "L9": "one small colorful singing flying tropical curious rare forest bird", # environment
        "L10": "one small colorful singing flying tropical curious rare forest social bird" # social behavior
    }
}
for key in diverse_dimension_description_dict3.keys():
    plot_random_descriptions(diverse_dimension_description_dict3[key]["L10"], n_sim_per_L=200, add_prefix=True)
# %%
social_identity_description_dict = {
    "person": {
        "L1": "one person",
        "L2": "one male person",                         # gender
        "L3": "one male Asian person",                   # race/ethnicity
        "L4": "one male Asian American person",          # nationality
        "L5": "one male Asian American middle-class person",  # socioeconomic class
        "L6": "one male Asian American middle-class Buddhist person",  # religion
        "L7": "one male Asian American middle-class Buddhist academic person",  # profession
        "L8": "one male Asian American middle-class Buddhist academic liberal person",  # ideology
        "L9": "one male Asian American middle-class Buddhist academic liberal millennial person",  # generation
        "L10": "one male Asian American middle-class Buddhist academic liberal millennial urban person"  # residence/region
    },
    "student": {
        "L1": "one student",
        "L2": "one female student",                      # gender
        "L3": "one female Black student",                # race
        "L4": "one female Black South-African student",  # nationality
        "L5": "one female Black South-African working-class student", # class
        "L6": "one female Black South-African working-class Christian student", # religion
        "L7": "one female Black South-African working-class Christian political-science student", # academic field
        "L8": "one female Black South-African working-class Christian political-science activist student", # ideology/role
        "L9": "one female Black South-African working-class Christian political-science activist Gen-Z student", # generation
        "L10": "one female Black South-African working-class Christian political-science activist Gen-Z campus-leader student" # social role
    },
    "worker": {
        "L1": "one worker",
        "L2": "one male worker",                         # gender
        "L3": "one male Hispanic worker",                # ethnicity
        "L4": "one male Hispanic Mexican worker",        # nationality
        "L5": "one male Hispanic Mexican blue-collar worker", # class
        "L6": "one male Hispanic Mexican blue-collar Catholic worker", # religion
        "L7": "one male Hispanic Mexican blue-collar Catholic factory worker", # occupation
        "L8": "one male Hispanic Mexican blue-collar Catholic factory unionized worker", # organization
        "L9": "one male Hispanic Mexican blue-collar Catholic factory unionized middle-aged worker", # generation
        "L10": "one male Hispanic Mexican blue-collar Catholic factory unionized middle-aged rural worker" # region
    },
    "teacher": {
        "L1": "one teacher",
        "L2": "one female teacher",                      # gender
        "L3": "one female White teacher",                # race
        "L4": "one female White British teacher",        # nationality
        "L5": "one female White British upper-middle-class teacher", # class
        "L6": "one female White British upper-middle-class secular teacher", # religion
        "L7": "one female White British upper-middle-class secular literature teacher", # field
        "L8": "one female White British upper-middle-class secular literature progressive teacher", # ideology
        "L9": "one female White British upper-middle-class secular literature progressive older teacher", # generation
        "L10": "one female White British upper-middle-class secular literature progressive older urban teacher" # region
    },
    "doctor": {
        "L1": "one doctor",
        "L2": "one male doctor",
        "L3": "one male South-Asian doctor",
        "L4": "one male South-Asian Indian doctor",
        "L5": "one male South-Asian Indian upper-class doctor",
        "L6": "one male South-Asian Indian upper-class Hindu doctor",
        "L7": "one male South-Asian Indian upper-class Hindu surgical doctor",
        "L8": "one male South-Asian Indian upper-class Hindu surgical conservative doctor",
        "L9": "one male South-Asian Indian upper-class Hindu surgical conservative middle-aged doctor",
        "L10": "one male South-Asian Indian upper-class Hindu surgical conservative middle-aged metropolitan doctor"
    },
    "artist": {
        "L1": "one artist",
        "L2": "one female artist",
        "L3": "one female East-Asian artist",
        "L4": "one female East-Asian Japanese artist",
        "L5": "one female East-Asian Japanese middle-class artist",
        "L6": "one female East-Asian Japanese middle-class non-religious artist",
        "L7": "one female East-Asian Japanese middle-class non-religious digital artist",
        "L8": "one female East-Asian Japanese middle-class non-religious digital feminist artist",
        "L9": "one female East-Asian Japanese middle-class non-religious digital feminist millennial artist",
        "L10": "one female East-Asian Japanese middle-class non-religious digital feminist millennial urban artist"
    },
    "soldier": {
        "L1": "one soldier",
        "L2": "one male soldier",
        "L3": "one male White soldier",
        "L4": "one male White American soldier",
        "L5": "one male White American working-class soldier",
        "L6": "one male White American working-class Christian soldier",
        "L7": "one male White American working-class Christian infantry soldier",
        "L8": "one male White American working-class Christian infantry patriotic soldier",
        "L9": "one male White American working-class Christian infantry patriotic veteran soldier",
        "L10": "one male White American working-class Christian infantry patriotic veteran rural soldier"
    },
    "politician": {
        "L1": "one politician",
        "L2": "one male politician",
        "L3": "one male Black politician",
        "L4": "one male Black American politician",
        "L5": "one male Black American upper-class politician",
        "L6": "one male Black American upper-class Christian politician",
        "L7": "one male Black American upper-class Christian democratic politician",
        "L8": "one male Black American upper-class Christian democratic progressive politician",
        "L9": "one male Black American upper-class Christian democratic progressive experienced politician",
        "L10": "one male Black American upper-class Christian democratic progressive experienced national politician"
    },
    "child": {
        "L1": "one child",
        "L2": "one young boy",                        # gender
        "L3": "one young Hispanic boy",               # ethnicity
        "L4": "one young Hispanic Brazilian boy",     # nationality
        "L5": "one young Hispanic Brazilian working-class boy", # class
        "L6": "one young Hispanic Brazilian working-class Catholic boy", # religion
        "L7": "one young Hispanic Brazilian working-class Catholic schoolboy", # role
        "L8": "one young Hispanic Brazilian working-class Catholic schoolboy playful boy", # trait
        "L9": "one young Hispanic Brazilian working-class Catholic schoolboy playful neighborhood boy", # context
        "L10": "one young Hispanic Brazilian working-class Catholic schoolboy playful neighborhood kind boy" # moral trait
    }
}
for key in social_identity_description_dict.keys():
    plot_random_descriptions(social_identity_description_dict[key]["L10"], n_sim_per_L=200, add_prefix=True)


#%%

slope_dict = {
"politician": {
        "L1": "one politician",
        "L2": "one male politician",
        "L3": "one male Black politician",
        "L4": "one male Black American politician",
        "L5": "one male Black American upper-class politician",
        "L6": "one male Black American upper-class Christian politician",
        "L7": "one male Black American upper-class Christian democratic politician",
        "L8": "one male Black American upper-class Christian democratic progressive politician",
        "L9": "one male Black American upper-class Christian democratic progressive experienced politician",
        "L10": "one male Black American upper-class Christian democratic progressive experienced national politician"
    },

    "book": {
        "L1": "one book",
        "L2": "one old book",                      # age
        "L3": "one old heavy book",                # weight
        "L4": "one old heavy illustrated book",    # content
        "L5": "one old heavy illustrated historical book", # subject
        "L6": "one old heavy illustrated historical leather-bound book", # material
        "L7": "one old heavy illustrated historical leather-bound handwritten book", # authenticity
        "L8": "one old heavy illustrated historical leather-bound handwritten religious book", # theme
        "L9": "one old heavy illustrated historical leather-bound handwritten religious rare book", # rarity
        "L10": "one old heavy illustrated historical leather-bound handwritten religious rare valuable book" # value
    },

    "street": {
        "L1": "one street",
        "L2": "one narrow street",                 # width
        "L3": "one narrow cobblestone street",     # material
        "L4": "one narrow cobblestone busy street", # activity
        "L5": "one narrow cobblestone busy market street", # function
        "L6": "one narrow cobblestone busy market colorful street", # visual
        "L7": "one narrow cobblestone busy market colorful European street", # geographic
        "L8": "one narrow cobblestone busy market colorful European festive street", # atmosphere
        "L9": "one narrow cobblestone busy market colorful European festive evening street", # time of day
        "L10": "one narrow cobblestone busy market colorful European festive evening rain-reflective street" # lighting/weather
    },

    "tree": {
        "L1": "one tree",
        "L2": "one tall tree",                     # size
        "L3": "one tall green tree",               # color
        "L4": "one tall green leafy tree",         # texture
        "L5": "one tall green leafy windy tree",   # dynamic condition
        "L6": "one tall green leafy windy riverside tree", # spatial context
        "L7": "one tall green leafy windy riverside ancient tree", # temporal
        "L8": "one tall green leafy windy riverside ancient twisted tree", # shape
        "L9": "one tall green leafy windy riverside ancient twisted sacred tree", # cultural meaning
        "L10": "one tall green leafy windy riverside ancient twisted sacred solitary tree" # social isolation / emotion
    },

    "car": {
        "L1": "one car",
        "L2": "one red car",                       # color
        "L3": "one red fast car",                  # speed/performance
        "L4": "one red fast quiet car",            # sound property
        "L5": "one red fast quiet electric car",   # mechanism
        "L6": "one red fast quiet electric compact car", # size
        "L7": "one red fast quiet electric compact city car", # usage context
        "L8": "one red fast quiet electric compact city rented car", # ownership
        "L9": "one red fast quiet electric compact city rented self-driving car", # function
        "L10": "one red fast quiet electric compact city rented self-driving futuristic car" # concept/futurity
    },

     "dog": {
        "L1": "one dog",
        "L2": "one brown dog",                     # color
        "L3": "one brown playful dog",             # temperament
        "L4": "one brown playful obedient dog",    # behavior trait
        "L5": "one brown playful obedient wet dog", # texture/state
        "L6": "one brown playful obedient wet running dog", # motion
        "L7": "one brown playful obedient wet running city dog", # location
        "L8": "one brown playful obedient wet running city park dog", # finer context
        "L9": "one brown playful obedient wet running city park curious dog", # cognition
        "L10": "one brown playful obedient wet running city park curious loyal dog" # moral trait
    },

    "table": {
        "L1": "one table",
        "L2": "one wooden table",
        "L3": "one wooden round table",
        "L4": "one wooden round polished table",
        "L5": "one wooden round polished dining table",
        "L6": "one wooden round polished dining sturdy table",
        "L7": "one wooden round polished dining sturdy minimalist table",
        "L8": "one wooden round polished dining sturdy minimalist light-brown table",
        "L9": "one wooden round polished dining sturdy minimalist light-brown clean table",
        "L10": "one wooden round polished dining sturdy minimalist light-brown clean elegant table"
    },

    "river": {
        "L1": "one river",
        "L2": "one calm river",
        "L3": "one calm clear river",
        "L4": "one calm clear flowing river",
        "L5": "one calm clear flowing narrow river",
        "L6": "one calm clear flowing narrow forest-side river",
        "L7": "one calm clear flowing narrow forest-side reflective river",
        "L8": "one calm clear flowing narrow forest-side reflective sunlit river",
        "L9": "one calm clear flowing narrow forest-side reflective sunlit tranquil river",
        "L10": "one calm clear flowing narrow forest-side reflective sunlit tranquil winding river"
    },

     "house": {
        "L1": "one house",
        "L2": "one small house",
        "L3": "one small wooden house",
        "L4": "one small wooden cozy house",
        "L5": "one small wooden cozy countryside house",
        "L6": "one small wooden cozy countryside light-blue house",
        "L7": "one small wooden cozy countryside light-blue porch-fronted house",
        "L8": "one small wooden cozy countryside light-blue porch-fronted well-kept house",
        "L9": "one small wooden cozy countryside light-blue porch-fronted well-kept flower-surrounded house",
        "L10": "one small wooden cozy countryside light-blue porch-fronted well-kept flower-surrounded charming house"
    },

    "tree": {
        "L1": "one tree",
        "L2": "one tall tree",
        "L3": "one tall green tree",
        "L4": "one tall green leafy tree",
        "L5": "one tall green leafy thick-trunked tree",
        "L6": "one tall green leafy thick-trunked sunlit tree",
        "L7": "one tall green leafy thick-trunked sunlit solitary tree",
        "L8": "one tall green leafy thick-trunked sunlit solitary hillside tree",
        "L9": "one tall green leafy thick-trunked sunlit solitary hillside wind-blown tree",
        "L10": "one tall green leafy thick-trunked sunlit solitary hillside wind-blown majestic tree"
    },

    "cup": {
        "L1": "one cup",
        "L2": "one ceramic cup",
        "L3": "one ceramic white cup",
        "L4": "one ceramic white small cup",
        "L5": "one ceramic white small round-edged cup",
        "L6": "one ceramic white small round-edged glossy cup",
        "L7": "one ceramic white small round-edged glossy heat-resistant cup",
        "L8": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped cup",
        "L9": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist cup",
        "L10": "one ceramic white small round-edged glossy heat-resistant smoothly-shaped minimalist lightweight cup"
        },

        "chair": {
        "L1": "one chair",
        "L2": "one wooden chair",
        "L3": "one wooden sturdy chair",
        "L4": "one wooden sturdy simple chair",
        "L5": "one wooden sturdy simple light-brown chair",
        "L6": "one wooden sturdy simple light-brown straight-backed chair",
        "L7": "one wooden sturdy simple light-brown straight-backed polished chair",
        "L8": "one wooden sturdy simple light-brown straight-backed polished minimalist chair",
        "L9": "one wooden sturdy simple light-brown straight-backed polished minimalist dining chair",
        "L10": "one wooden sturdy simple light-brown straight-backed polished minimalist dining smooth-edged chair"
        },

        "car": {
        "L1": "one car",
        "L2": "one red car",
        "L3": "one red compact car",
        "L4": "one red compact modern car",
        "L5": "one red compact modern electric car",
        "L6": "one red compact modern electric aerodynamic car",
        "L7": "one red compact modern electric aerodynamic sleek car",
        "L8": "one red compact modern electric aerodynamic sleek glossy car",
        "L9": "one red compact modern electric aerodynamic sleek glossy two-door car",
        "L10": "one red compact modern electric aerodynamic sleek glossy two-door lightweight car"
    }
}
for key in slope_dict.keys():
    plot_random_descriptions(slope_dict[key]["L10"], n_sim_per_L=30, add_prefix=True)
# %% helper function
# 
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine

def extract_description_words(sentence, template):
    """
    Extract description words from sentence based on template.
    
    Args:
        sentence: e.g., "one wooden round polished dining sturdy minimalist light-brown clean elegant table"
        template: e.g., "one__table"
    
    Returns:
        list of description words
    """
    # Escape special regex characters and replace __ with capture group
    pattern = template.replace("__", "(.*?)")
    pattern = "^" + pattern + "$"
    
    match = re.match(pattern, sentence)
    if match:
        description_part = match.group(1).strip()
        description_word_list = description_part.split()
        return description_word_list
    else:
        return []

def plot_description_word_similarity_heatmap(sentence, template, model):
    """
    Plot cosine similarity heatmap between description words with norm visualization.
    
    Args:
        sentence: input sentence
        template: template pattern like "one__table"
        model: sentence transformer model
    """
    # Extract description words
    description_word_list = extract_description_words(sentence, template)
    
    if len(description_word_list) == 0:
        print("No description words found!")
        return
    
    # Get embeddings for each word
    embeddings = extract_embedding(description_word_list, type="text", normalize=False)
    
    # Calculate norms
    norms = embeddings.norm(dim=-1).cpu().numpy()
    
    # Calculate cosine similarity matrix
    n_words = len(description_word_list)
    similarity_matrix = np.zeros((n_words, n_words))
    
    for i in range(n_words):
        for j in range(n_words):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])
    
    # Create figure with two subplots
    fig, (ax_norm, ax_heatmap) = plt.subplots(1, 2, figsize=(14, 10), 
                                               gridspec_kw={'width_ratios': [1, 10]})
    
    # Plot norm column
    norm_matrix = norms.reshape(-1, 1)
    sns.heatmap(norm_matrix, annot=True, fmt='.3f', cmap='Greys', 
                cbar=False, ax=ax_norm, yticklabels=description_word_list,
                xticklabels=['Norm'])
    ax_norm.set_title('Embedding Norm', fontsize=12)
    
    # Plot similarity heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=description_word_list, yticklabels=description_word_list,
                vmin=0, vmax=1, ax=ax_heatmap, cbar_kws={'label': 'Cosine Similarity'})
    ax_heatmap.set_title(f'Cosine Similarity Heatmap\nSentence: "{sentence}"', fontsize=12)
    ax_heatmap.set_xlabel('Words', fontsize=10)
    ax_heatmap.set_ylabel('Words', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return description_word_list, embeddings, norms, similarity_matrix


# %%
test_sentence = "wooden round polished dining sturdy minimalist light-brown clean elegant"
template = "__"
plot_random_descriptions(test_sentence, n_sim_per_L=30, add_prefix=True)

description_words, embeddings, norms, sim_matrix = plot_description_word_similarity_heatmap(
    test_sentence, template, model
)

# %%
from random import sample
def plot_word_inclusion_vs_norm(sentence, template, model, n_samples=30, max_k=10):
    """
    Given a sentence and template, extract description words and analyze how including
    each word affects the norm of the embedding.
    
    For each word, creates a scatter plot showing:
    - X-axis: number of descriptions (k)
    - Y-axis: embedding norm
    - Red points: sentences that include this word
    - Gray points: sentences that don't include this word
    
    Args:
        sentence: String containing description words
        template: Template string with "__" placeholder
        model: The embedding model
        n_samples: Number of random samples to generate for each k
        max_k: Maximum number of words to sample
    """
    # Extract description words
    # Remove template words from sentence first
    template_words = [word.strip() for word in template.replace("__", " ").split() if word.strip()]
    description_words = [word.strip() for word in sentence.split() if word.strip() and word not in template_words]
    n_words = len(description_words)
    
    if max_k > n_words:
        max_k = n_words
    
    # For each word, collect data points
    word_data = {word: {'with_word': {'k': [], 'norm': []}, 
                        'without_word': {'k': [], 'norm': []}} 
                 for word in description_words}
    
    # Generate samples for each k
    for k in range(1, max_k + 1):
        for _ in range(n_samples):
            # Randomly sample k words without replacement
            sampled_words = sample(description_words, k)
            sampled_sentence = " ".join(sampled_words)
            full_sentence = template.replace("__", sampled_sentence)
            
            # Get embedding and norm
            embedding = extract_embedding([full_sentence], type="text", normalize=False)
            norm = embedding.norm(dim=-1).cpu().numpy()[0]
            
            # Record for each word whether it was included
            for word in description_words:
                if word in sampled_words:
                    word_data[word]['with_word']['k'].append(k)
                    word_data[word]['with_word']['norm'].append(norm)
                else:
                    word_data[word]['without_word']['k'].append(k)
                    word_data[word]['without_word']['norm'].append(norm)
    
    # Create subplots for each word
    n_cols = 3
    n_rows = (n_words + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_words > 1 else [axes]
    
    for idx, word in enumerate(description_words):
        ax = axes[idx]
        
        # Plot points without the word (gray)
        ax.scatter(word_data[word]['without_word']['k'], 
                  word_data[word]['without_word']['norm'],
                  c='gray', alpha=0.5, s=20, label='Without word')
        
        # Plot points with the word (red)
        ax.scatter(word_data[word]['with_word']['k'], 
                  word_data[word]['with_word']['norm'],
                  c='red', alpha=0.6, s=20, label='With word')
        
        ax.set_xlabel('Number of descriptions (k)', fontsize=10)
        ax.set_ylabel('Embedding norm', fontsize=10)
        ax.set_title(f'Word: "{word}"', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_words, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Word Inclusion vs Embedding Norm\nTemplate: "{template}"', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return word_data

# %%
def plot_word_norm_vs_correlation(test_sentence, template, model, n_samples=50, max_k=9):
    """
    Plot relationship between word norm and average correlation with other words.
    Each point represents a description word.
    X-axis: word embedding norm
    Y-axis: average correlation with other words
    Color: average norm of sentences containing the word
    Size: variance of norm of sentences containing the word
    """
    # Get description words and their embeddings
    description_words = extract_description_words(test_sentence, template)
    n_words = len(description_words)
    
    # Get word embeddings and norms
    word_embeddings = {}
    word_norms = {}
    for word in description_words:
        emb = extract_embedding([word], type="text", normalize=False)
        word_embeddings[word] = emb
        word_norms[word] = emb.norm(dim=-1).cpu().numpy()[0]
    
    # Calculate correlation matrix between words
    embeddings_list = [word_embeddings[word] for word in description_words]
    embeddings_tensor = torch.cat(embeddings_list, dim=0).cpu().numpy()
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = embeddings_tensor / np.linalg.norm(embeddings_tensor, axis=1, keepdims=True)
    correlation_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    
    # Calculate average correlation for each word (excluding self-correlation)
    avg_correlations = {}
    for i, word in enumerate(description_words):
        # Exclude diagonal (self-correlation)
        correlations = np.concatenate([correlation_matrix[i, :i], correlation_matrix[i, i+1:]])
        avg_correlations[word] = np.mean(correlations)
    
    # Calculate sentence norms with each word
    word_sentence_stats = {}
    
    for word in description_words:
        with_word_norms = []
        
        for _ in range(n_samples):
            k = np.random.randint(1, max_k)
            sampled_words = sample(description_words, k)
            
            # Create description
            description = template.replace('__', ', '.join(sampled_words))
            
            # Get embedding and norm
            embedding = extract_embedding([description], type="text", normalize=False)
            norm = embedding.norm(dim=-1).cpu().numpy()[0]
            
            if word in sampled_words:
                with_word_norms.append(norm)
        
        # Calculate statistics for sentences containing the word
        word_sentence_stats[word] = {
            'mean_norm': np.mean(with_word_norms) if with_word_norms else 0,
            'var_norm': np.std(with_word_norms) if with_word_norms else 0
        }
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_values = [word_norms[word] for word in description_words]
    y_values = [avg_correlations[word] for word in description_words]
    colors = [word_sentence_stats[word]['mean_norm'] for word in description_words]
    sizes = [word_sentence_stats[word]['var_norm'] * 1000 for word in description_words]  # Scale for visibility
    
    scatter = ax.scatter(x_values, y_values, c=colors, s=sizes, 
                        alpha=0.6, cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add word labels
    for i, word in enumerate(description_words):
        ax.annotate(word, (x_values[i], y_values[i]), 
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')
    
    # Add colorbar for mean sentence norm
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Average Sentence Norm (with word)', fontsize=11)
    
    ax.set_xlabel('Word Embedding Norm', fontsize=12)
    ax.set_ylabel('Average Correlation with Other Words', fontsize=12)
    ax.set_title(f'Word Norm vs Correlation\nTemplate: "{template}"\n(Color = Mean Norm, Size = Variance of Sentences with Word)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return word_norms, avg_correlations, word_sentence_stats

# %%


no_slope_dict = {
    "table": {
        "test_sentence": "one elegant cheerful determined artistic traveling modern red-dressed confident metropolitan woman",
        "template": "one__woman"
    },
    "man": {
        "test_sentence": "one tall smiling tired office walking polite bearded thoughtful urban man",
        "template": "one__man"
    },
    "cat": {
        "test_sentence": "one white fluffy calm small sleeping window-side domestic content purring cat",
        "template": "one__cat"
    },
    "tree": {
        "test_sentence": "one tall green leafy thick-trunked sunlit solitary hillside wind-blown majestic tree",
        "template": "one__tree"
    }
}


slope_dict = {
    "table": {
        "test_sentence": "one wooden round polished dining sturdy minimalist light-brown clean elegant table",
        "template": "one__table"
    },
    "politician": {
        "test_sentence": "one male Black American upper-class Christian democratic progressive experienced national politician",
        "template": "one__politician"
    },
    "car": {
        "test_sentence": "one red compact modern electric aerodynamic sleek glossy two-door lightweight car",
        "template": "one__car"
    },
    "chair": {
        "test_sentence": "one wooden sturdy simple light-brown straight-backed polished minimalist dining smooth-edged chair",
        "template": "one__chair"
    }
}

#%%
sentence_dict = slope_dict
for key in sentence_dict.keys():
    description_words, embeddings, norms, sim_matrix = plot_description_word_similarity_heatmap(
        sentence_dict[key]["test_sentence"], sentence_dict[key]["template"], model
    )
    word_norms, avg_correlations, word_sentence_stats = plot_word_norm_vs_correlation(
        sentence_dict[key]["test_sentence"], 
        sentence_dict[key]["template"], 
        model, 
        n_samples=50, 
        max_k=9
    )
    _=plot_word_inclusion_vs_norm(sentence_dict[key]["test_sentence"], sentence_dict[key]["template"], model, n_samples=50, max_k=9)

# %%
