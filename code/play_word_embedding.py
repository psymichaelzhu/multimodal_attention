# %% objective
# obtain text or image embedding from CLIP

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
os.chdir('/Users/rezek_zhu/multimodal_attention')  
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

# save directory
save_dir = 'data/social_sentence'
os.makedirs(save_dir, exist_ok=True)

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

# %% prepare sentence list
# Three dimensions describing social interaction: who, what, where

# 1. Define each dimension as a list of 20 strings (including 5 uncommon ones)
who_list = [
    "parent-child", "teacher-student", "siblings", "friends", "colleagues",
    "doctor-patient", "coach-player", "customer-cashier", "neighbor-neighbor", "boss-employee",
    "grandparent-grandchild", "partner-partner", "mentor-mentee", "tour guide-tourist", "parent-babysitter",
    "volunteer-beneficiary", "artist-audience", "referee-athlete", "host-guest", "scientist-participant"
]

what_list = [
    "play", "discuss", "eat", "read", "study", 
    "argue", "help", "teach", "celebrate", "share",
    "shop", "compete", "comfort", "guide", "observe",
    "debate", "negotiate", "complain", "explore", "perform"
]

where_list = [
    "living room", "playground", "school", "office", "park",
    "restaurant", "hospital", "classroom", "stadium", "supermarket",
    "theater", "laboratory", "library", "museum", "backyard",
    "studio", "conference room", "zoo", "airport", "garden"
]

# 2. Combine into sentence list Who+what+where, e.g. "Parent-child play living room"
sentence_list = []
for who in who_list:
    for what in what_list:
        for where in where_list:
            sentence = f"{who} {what} {where}"
            sentence_list.append(sentence)


#%% extract text embedding
# You may want to batch if too large (for efficiency and memory, do e.g. by 5000)
batch_size = 50
text_embeddings = []
for i in tqdm(range(0, len(sentence_list), batch_size)):
    batch_sent = sentence_list[i:i+batch_size]
    batch_emb = extract_embedding(batch_sent, type="text")
    text_embeddings.append(batch_emb.cpu().numpy())
text_embeddings = np.concatenate(text_embeddings, axis=0)

print(f"Generated {len(sentence_list)} sentences. Embedding shape: {text_embeddings.shape}")
# save embeddings
embedding_dir = os.path.join(save_dir, 'embedding/{}/{}/{}'.format(model_source.replace('/', '-'), model_name.replace('/', '-'), pretrained_name.replace('/', '-')))
os.makedirs(embedding_dir, exist_ok=True)
np.save(os.path.join(embedding_dir, 'sentence_embedding.npy'), text_embeddings)

# %% similarity analysis
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# ---- Fixed limits for all plots ----
# For cosine similarity, [-1, 1] is max range but 0~1 expected here for mean embeddings (can tune if needed)
SIM_HEATMAP_VMIN = 0.5
SIM_HEATMAP_VMAX = 1.0
PAIRWISE_SCATTER_YLIM = (0.0, 1.05)
CORRELATION_XLIM = (0.6, 0.85)
CORRELATION_YLIM = (0.7, 0.9)
PAIRWISE_SCATTER_ALPHA = 0.15
PAIRWISE_SCATTER_JITTER = 0.18

def plot_dimension_similarity(text_embeddings, who_list, what_list, where_list, 
                             dimension='who', sample_size=10, random_seed=42, return_corr_data=False):
    """
    Compute and visualize within-group and between-group similarities for a specified dimension.
    Supported dimensions: 'who', 'what', 'where'.
    Plots:
        - Between-group similarity heatmap (sampled)
        - Within-group similarity scatter (sampled) with fixed axes and jitter
        - Correlation between within-group and between-group mean similarity (fixed axes)
    If return_corr_data is True, also return (within_means, between_means, dimension)
    """
    # Prepare group lists and indices
    if dimension == 'who':
        group_list = who_list
        group_num = len(who_list)
        lvlA_num = len(who_list)
        lvlB_num = len(what_list)
        lvlC_num = len(where_list)
        group2idx = {}
        for i, name in enumerate(group_list):
            group2idx[name] = list(range(i * lvlB_num * lvlC_num, (i+1)*lvlB_num*lvlC_num))
    elif dimension == 'what':
        group_list = what_list
        group_num = len(what_list)
        lvlA_num = len(who_list)
        lvlB_num = len(what_list)
        lvlC_num = len(where_list)
        group2idx = {}
        for j, name in enumerate(group_list):
            idxs = []
            for i in range(lvlA_num):
                for k in range(lvlC_num):
                    idx = i * lvlB_num * lvlC_num + j * lvlC_num + k
                    idxs.append(idx)
            group2idx[name] = idxs
    elif dimension == 'where':
        group_list = where_list
        group_num = len(where_list)
        lvlA_num = len(who_list)
        lvlB_num = len(what_list)
        lvlC_num = len(where_list)
        group2idx = {}
        for k, name in enumerate(group_list):
            idxs = []
            for i in range(lvlA_num):
                for j in range(lvlB_num):
                    idx = i * lvlB_num * lvlC_num + j * lvlC_num + k
                    idxs.append(idx)
            group2idx[name] = idxs
    else:
        raise ValueError("dimension must be one of ['who', 'what', 'where']")

    # 1. Compute mean embedding for each group
    group_means = []
    for name in group_list:
        idxs = group2idx[name]
        emb = text_embeddings[idxs]
        mean_emb = emb.mean(axis=0)
        group_means.append(mean_emb)
    group_means = np.stack(group_means)  # (n_group, emb_dim)

    # 2. Compute between-group similarity matrix
    group_sims = cosine_similarity(group_means)

    # 3. Compute within-group similarities (all pairwise in group, exclude diagonals)
    within_group_sims = []
    for name in group_list:
        idxs = group2idx[name]
        emb = text_embeddings[idxs]
        sim_matrix = cosine_similarity(emb)
        triu_idx = np.triu_indices_from(sim_matrix, k=1)
        sim_scores = sim_matrix[triu_idx]
        within_group_sims.append(sim_scores)

    # 4. Randomly sample group indices for plotting
    random.seed(random_seed)
    sample_size = min(sample_size, group_num)
    sampled = random.sample(range(group_num), sample_size)
    sampled_names = [group_list[i] for i in sampled]

    plt.figure(figsize=(14,6))

    # (a) Between-group similarity heatmap
    plt.subplot(1,2,1)
    submat = group_sims[np.ix_(sampled, sampled)]
    sns.heatmap(
        submat, 
        xticklabels=sampled_names, yticklabels=sampled_names, 
        annot=True, fmt=".2f", 
        cmap="viridis", vmin=SIM_HEATMAP_VMIN, vmax=SIM_HEATMAP_VMAX, 
        cbar_kws={'label':'Cosine sim.', 'shrink': 0.95}
    )
    plt.title(f"Between-{dimension} mean embedding similarity ({sample_size} sampled {dimension})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # (b) Within-group similarity scatter with jitter and fixed y limits
    plt.subplot(1,2,2)
    for scatter_no, idx in enumerate(sampled):
        sim_scores = within_group_sims[idx]
        # Jitter: add small noise to the x-position to reduce overlap
        x_base = scatter_no  # integer x corresponding to the category
        x_jittered = x_base + np.random.uniform(-PAIRWISE_SCATTER_JITTER, PAIRWISE_SCATTER_JITTER, size=len(sim_scores))
        plt.scatter(x_jittered, sim_scores, alpha=PAIRWISE_SCATTER_ALPHA, color='C0', s=22, edgecolor='none')
    plt.xticks(range(sample_size), sampled_names, rotation=90)
    plt.ylabel("Pairwise similarity")
    plt.ylim(PAIRWISE_SCATTER_YLIM)
    plt.title(f"Within-{dimension} pairwise similarity ({sample_size} sampled {dimension})")
    plt.tight_layout()
    plt.gca().set_xlim(-0.5, sample_size-0.5)
    plt.show()

    # 5. Compute groupwise means
    within_means = np.array([np.mean(scores) for scores in within_group_sims])  # (n_group,)
    between_means = []
    for i in range(group_num):
        sims = group_sims[i]
        mean_btwn = (np.sum(sims) - sims[i]) / (group_num - 1)
        between_means.append(mean_btwn)
    between_means = np.array(between_means)

    # 6. Plot correlation with fixed axes
    plt.figure(figsize=(6,6))
    ax = sns.regplot(
        x=within_means, y=between_means, ci=95, scatter_kws={'s':30, 'alpha':PAIRWISE_SCATTER_ALPHA, 'color':'C0'})
    plt.xlabel(f"Within-{dimension} mean similarity")
    plt.ylabel(f"Between-{dimension} mean similarity to others")
    plt.title(f"Correlation: Within-{dimension} vs. Between-{dimension} similarity")
    plt.xlim(CORRELATION_XLIM)
    plt.ylim(CORRELATION_YLIM)
    plt.grid(True)

    # Pearson correlation annotation (bottom right)
    corr, pval = pearsonr(within_means, between_means)
    plt.text(
        CORRELATION_XLIM[1], CORRELATION_YLIM[0], 
        f"r={corr:.2f}, p={pval:.3g}", 
        horizontalalignment='right', verticalalignment='bottom'
    )
    plt.tight_layout()
    plt.show()

    # --- 新增: 如果要求，返回最后一张图的两组数据及标签 ---
    if return_corr_data:
        return within_means, between_means, dimension
    else:
        return None

# Example Usage:
corr_data = {}  # 用于保存每个维度的最后一张图数据
for dim in ['who', 'what', 'where']:
    result = plot_dimension_similarity(
        text_embeddings, who_list, what_list, where_list, 
        dimension=dim, 
        sample_size=10, 
        random_seed=42,
        return_corr_data=True
    )
    if result is not None:
        within_means, between_means, dimension_label = result
        corr_data[dimension_label] = (within_means, between_means)

# --- 拼合所有维度的correlation数据到一张图 ---
plt.figure(figsize=(8,6))
dim_colors = {'who': 'C0', 'what': 'C1', 'where': 'C2'}
for dim in ['who', 'what', 'where']:
    if dim in corr_data:
        within_means, between_means = corr_data[dim]
        plt.scatter(
            within_means, between_means, 
            alpha=0.30, s=38, 
            label=dim, color=dim_colors[dim]
        )
        # Pearson 相关系数
        corr, pval = pearsonr(within_means, between_means)
        plt.text(
            CORRELATION_XLIM[1]-0.02, 
            CORRELATION_YLIM[0]+0.025+0.03*(['who','what','where'].index(dim)), 
            f"{dim}: r={corr:.2f}, p={pval:.3g}", 
            color=dim_colors[dim],
            fontsize=11,
            horizontalalignment='right',
            verticalalignment='bottom'
        )
plt.xlabel("Within-group mean similarity")
plt.ylabel("Between-group mean similarity to others")
plt.title("Correlation: Within vs. Between Similarity (All Dimensions)")
plt.xlim(CORRELATION_XLIM)
plt.ylim(CORRELATION_YLIM)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
override, reading