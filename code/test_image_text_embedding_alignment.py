# %% objective
# validate the alignment of image and text embedding in CLIP model
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
def extract_embedding(objects, type="image",normalize=True):
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
base_dir = 'data/image_text_alignment'
image_dir = os.path.join(base_dir, 'picture')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# load descriptions from description.txt
description_path = os.path.join(base_dir, 'description.txt')
description_dict = {}
with open(description_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',', 1)  # split only on first comma
            if len(parts) == 2:
                image_name, description = parts
                description_dict[image_name.strip()] = description.strip()

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
        
        # only include if we have a description for this image
        if image_name in description_dict:
            images.append(image)
            image_names.append(image_name)
            descriptions.append(f"a picture of {description_dict[image_name]}")
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

#%% extract embeddings
# extract image embeddings
image_embeddings = extract_embedding(images, type="image")

# extract text embeddings from descriptions
text_embeddings = extract_embedding(descriptions, type="text")

# normalize embeddings for cosine similarity
image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


#%% plot correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns
def plot_similarity_heatmap(embeddings1, embeddings2, labels1, labels2, title, xlabel, ylabel, figsize=(10, 8), scale_to_01=True):
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

# plot image-text similarity heatmap
plot_similarity_heatmap(text_embeddings_norm, image_embeddings_norm, 
                        image_names, image_names,
                        'Image-Text Embedding Alignment Heatmap',
                        'Images', 'Text Descriptions')

# plot within-text similarity heatmap
plot_similarity_heatmap(text_embeddings_norm, text_embeddings_norm,
                        image_names, image_names,
                        'Within-Text Embedding Similarity Heatmap',
                        'Text Descriptions', 'Text Descriptions')

# plot within-image similarity heatmap
plot_similarity_heatmap(image_embeddings_norm, image_embeddings_norm,
                        image_names, image_names,
                        'Within-Image Embedding Similarity Heatmap',
                        'Images', 'Images')

# %%
# conclusion
'''Similar (relative) patterns in within-text, within-image, image-text heatmaps;
yet the image has in general higher similarities; text is the largest contrast; image-text is weak, but pattern StopAsyncIteration
Could this be explained by the principle? image as text + modality, thus becomes more similar.
Inherently images have more dimensions to be similar, thus the norm of shared dimension increases, leading to higher similarities.
'''

# images have longer norms, encoding more information, leading to be more similar in absolute terms    
#%%
# compute the correlation between the norms of image and text embeddings
image_norms = image_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
text_norms = text_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
correlation = np.corrcoef(image_norms, text_norms)[0, 1]
print(f"Correlation between image and text norms: {correlation:.4f}")

#%%
# compute the correlation between the norms of image and text embeddings
image_norms = image_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
text_norms = text_embeddings.norm(dim=-1, keepdim=True).cpu().numpy()
correlation = np.corrcoef(image_norms, text_norms)[0, 1]
print(f"Correlation between image and text norms: {correlation:.4f}")