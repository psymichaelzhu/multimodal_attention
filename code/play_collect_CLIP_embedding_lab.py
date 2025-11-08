# %% objective
# collect embeddings from CLIP model for lab stimulus pictures

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
model_rank = [
    ('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b'),
    ('openclip', 'ViT-H-14-quickgelu', 'dfn5b'),
    ('openclip', 'ViT-bigG-14-quickgelu', 'metaclip_fullcc'),
    ('clip', 'ViT-L/14@336px', 'openai'),
    ('clip', 'ViT-B/32', 'openai'),
    ('openclip', 'ViT-bigG-14-CLIPA-336', 'datacomp1b')
]
model_rank_index = 0
model_source, model_name, pretrained_name = model_rank[model_rank_index]
print(model_source, model_name, pretrained_name)

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
# embedding directory for lab stimuli pictures
embedding_dir = 'data/lab_stimuli_color/embedding/{}/{}/{}'.format(
    model_source.replace('/', '-'),
    model_name.replace('/', '-'),
    pretrained_name.replace('/', '-')
)
print("embedding_dir:", embedding_dir)
os.makedirs(embedding_dir, exist_ok=True)

# image list
picture_dir = 'data/lab_stimuli_color/picture'
picture_name_list = [f for f in os.listdir(picture_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

embedding_path = os.path.join(embedding_dir, 'picture_embedding.npy')
name_path = os.path.join(embedding_dir, 'picture_name_list.npy')
overwrite = False

if os.path.exists(embedding_path) and not overwrite:
    print("Loading existing embeddings ...")    
    picture_embedding = torch.from_numpy(np.load(embedding_path)).to(device)
else:
    images = []
    for picture_name in tqdm(picture_name_list, desc="Loading images"):
        img_path = os.path.join(picture_dir, picture_name)
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            images.append(Image.new('RGB', (224, 224), color=(255, 255, 255)))  # fallback white image

    print("Extracting embeddings ...")
    # batched processing if necessary
    batch_size = 64
    all_embeddings = []
    model = model.to(device)
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing embeddings"):
            batch_imgs = images[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            batch_emb = model.encode_image(batch_tensors)
            all_embeddings.append(batch_emb.cpu())
    picture_embedding = torch.cat(all_embeddings, dim=0)
    # Save embeddings and names
    np.save(embedding_path, picture_embedding.numpy())
    np.save(name_path, np.array(picture_name_list))

print("Embedding shape:", picture_embedding.shape)
print("Number of pictures:", len(picture_name_list))

# %%
