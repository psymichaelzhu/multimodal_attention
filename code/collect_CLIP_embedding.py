# %% objective
# collect embeddings from CLIP model
# %% preparation

# packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from tqdm import tqdm

# set working directory to code directory
os.chdir('/Users/rezek_zhu/clip_social_annotation/code')  
print(os.getcwd())

# model info
model_rank = [('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b'),
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
    #print("available models:", open_clip.list_pretrained())
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
    model.eval() # switch to evaluation mode
    tokenizer = open_clip.get_tokenizer(model_name)
elif model_source == "clip":
    #print("available models:", clip.available_models())
    import clip
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize

# loading settings
overwrite = False
load_frame = True


# %% helper function
def extract_embedding(objects, type="image"):
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
    return features
    
def get_first_frame(video_path):
    """
    Extract the first frame from a video file and resize it to standard dimensions (300x250)

    input:
        video_path (str): Path to the video file
    output:
        frame (numpy array): First frame of the video in RGB format resized to standard dimensions, None if failed
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # resize to standard dimensions
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 250))
        return frame
    return None

# %% video embedding
# video and frame embedding directory
embedding_dir = '../data/embedding/{}/{}/{}'.format(model_source.replace('/', '-'), model_name.replace('/', '-'), pretrained_name.replace('/', '-'))
print("embedding_dir:", embedding_dir)
os.makedirs(embedding_dir, exist_ok=True)

# video list
video_dir = '../data/video/original_clips'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
#video_name_list = video_name_list[:2]#test

if os.path.exists(os.path.join(embedding_dir, 'video_embedding.npy')) and not overwrite:
    # Load existing embeddings
    print("Loading existing embeddings ...")    
    video_embedding = torch.from_numpy(np.load(os.path.join(embedding_dir, 'video_embedding.npy'))).to(device)
    if load_frame:
        frame_embeddings = np.load(os.path.join(embedding_dir, 'frame_embedding.npy'), allow_pickle=True)
        frame_embeddings = [emb.to(device) if isinstance(emb, torch.Tensor) else torch.from_numpy(emb).to(device) for emb in frame_embeddings]
    else:
        frame_embeddings = None
else:
    # store frame embeddings for each video
    frame_embeddings = []  # list of frame embeddings for each video
    checkpoint_interval = 30  # save every 30 videos
    
    for idx, video_name in enumerate(tqdm(video_name_list, desc="Processing videos")):
        # frame extraction
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # frame embedding
        frame_embedding = extract_embedding(frames, type="image")
        frame_embeddings.append(frame_embedding)

        # save checkpoint every 30 videos
        if (idx + 1) % checkpoint_interval == 0:
            print(f"\nSaving checkpoint at video {idx + 1}/{len(video_name_list)}...")
            
            # compute video embeddings for processed videos
            video_embedding_checkpoint = torch.zeros((len(frame_embeddings), model.visual.output_dim)).to(device)
            for i, emb in enumerate(frame_embeddings):
                video_embedding_checkpoint[i] = torch.mean(emb, dim=0)
                
            # save checkpoints
            np.save(os.path.join(embedding_dir, f'video_embedding_checkpoint_{idx+1}.npy'), 
                   video_embedding_checkpoint.cpu().numpy())
            np.save(os.path.join(embedding_dir, f'frame_embedding_checkpoint_{idx+1}.npy'), 
                   np.array(frame_embeddings, dtype=object))

    # compute final video embeddings
    video_embedding = torch.zeros((len(video_name_list), model.visual.output_dim)).to(device)
    for i, frame_embedding in enumerate(frame_embeddings):
        video_embedding[i] = torch.mean(frame_embedding, dim=0)

    # save final video and frame embeddings
    print("\nSaving final embeddings...")
    np.save(os.path.join(embedding_dir, 'video_embedding.npy'), video_embedding.cpu().numpy())
    np.save(os.path.join(embedding_dir, 'frame_embedding.npy'), np.array(frame_embeddings, dtype=object))

# video embedding matrix: video by embedding_dim
print(video_embedding.shape)
if frame_embeddings is not None:
    print(len(frame_embeddings))
    print(frame_embeddings[0].shape)






