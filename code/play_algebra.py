# %% objective
# Check the relationship between image embedding and the sum of who and where embedding

# %% preparation    
import os
import re
from PIL import Image
import numpy as np

os.chdir('/Users/rezek_zhu/multimodal_attention')
print(os.getcwd())

picture_dir = os.path.join('data/genAI_stimuli2', 'picture')
embedding_dir = os.path.join('data/genAI_stimuli2', 'embedding')

# %% 1. Load picture filenames and embeddings
image_files = [
    f for f in os.listdir(picture_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
]


embedding_path = os.path.join(embedding_dir, 'picture_embedding.npy')
name_list_path = os.path.join(embedding_dir, 'picture_name_list.npy')

picture_embeddings = np.load(embedding_path)  # shape: (N, D)
picture_name_list = np.load(name_list_path, allow_pickle=True)  # shape: (N,)

name_to_index = {name: idx for idx, name in enumerate(picture_name_list)}

def load_embedding(name):
    idx = name_to_index.get(name)
    if idx is None:
        print(f"Embedding not found for {name}")
        return None
    return picture_embeddings[idx]


# image embedding dict
image_emb_dict = {}
for name in picture_name_list:
    emb = load_embedding(name)
    if emb is not None:
        image_emb_dict[name] = emb

# Example usage:
print(image_emb_dict)

# %% 2. obtain who and where embedding
def remove_digits(s):
    return re.sub(r'\d+', '', s)

who_set = set()
where_set = set()
image_info_list = []

for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    parts = base_name.split('_')
    parts = [remove_digits(part) for part in parts]
    if len(parts) != 2:
        continue  
    who, where = parts
    who_set.add(who)
    where_set.add(where)
    image_info_list.append({
        'file': image_file,
        'basename': base_name,
        'who': who,
        'where': where
    })

print(who_set)
print(where_set)

# %% Generate CLIP embedding for each 'who' and 'where' concept

import torch
from PIL import Image

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

# Get sorted unique lists for reproducibility
who_list = sorted(list(who_set))
where_list = sorted(list(where_set))

# Compute who embeddings using torch.no_grad() so no grad is required
with torch.no_grad():
    who_emb_arr = model.encode_text(tokenizer(who_list).to(device))
    where_emb_arr = model.encode_text(tokenizer(where_list).to(device))

# Turn into dicts for lookup, safely detaching
who_emb_dict = {who: who_emb_arr[i].cpu().detach().numpy() for i, who in enumerate(who_list)}
where_emb_dict = {where: where_emb_arr[i].cpu().detach().numpy() for i, where in enumerate(where_list)}

#%% embedding of combination
base_name_list = sorted({info['basename'] for info in image_info_list})
base_emb_dict = {}
with torch.no_grad():
    base_emb_arr = model.encode_text(tokenizer(base_name_list).to(device))
    for i, base_name in enumerate(base_name_list):
        base_emb_dict[base_name] = base_emb_arr[i].cpu().detach().numpy()



# %% 5. check the relationship between combination embedding and the sum of component embeddings

# algebra only stands for the same modality

# 5-1 text
results = []

for info in image_info_list:
    image_emb = base_emb_dict.get(info['basename'])
    who_emb = who_emb_dict.get(info['who'])
    where_emb = where_emb_dict.get(info['where'])
    if image_emb is not None and who_emb is not None and where_emb is not None:
        algebra_emb = who_emb + where_emb
        if np.std(image_emb) > 0 and np.std(algebra_emb) > 0:
            similarity = np.corrcoef(image_emb, algebra_emb)[0, 1]
        else:
            similarity = float('nan')
        results.append({
            'image': info['basename'],
            'who': info['who'],
            'where': info['where'],
            'similarity': similarity
        })


print("Algebraic embedding similarities:")
for r in results[:10]:
    print(f"{r['image']}: who={r['who']}, where={r['where']}, similarity={r['similarity']:.4f}")


# %% 5-2 image
# embedding algebra (king - queen = man - woman) stands for image as well
results = []

for info in image_info_list:
    image_emb = image_emb_dict.get(info['file'])
    if 'none' in info['file']:
        continue
    who_image_emb = image_emb_dict.get(info['who'] + '_none.png')
    where_image_emb = image_emb_dict.get("none_" + info['where'] + '.png')
    
    if image_emb is not None and who_image_emb is not None and where_image_emb is not None:
        algebra_emb = who_image_emb + where_image_emb
        if np.std(image_emb) > 0 and np.std(algebra_emb) > 0:
            similarity = np.corrcoef(image_emb, algebra_emb)[0, 1]
        else:
            similarity = float('nan')
        results.append({
            'image': info['basename'],
            'who': info['who'],
            'where': info['where'],
            'similarity': similarity
        })

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        # who image
        who_img_path = os.path.join(picture_dir, info['who'] + '_none.png')
        who_img = plt.imread(who_img_path)
        axes[0].imshow(who_img)
        axes[0].set_title('who')
        axes[0].axis('off')
        # where image
        where_img_path = os.path.join(picture_dir, 'none_' + info['where'] + '.png')
        where_img = plt.imread(where_img_path)
        axes[1].imshow(where_img)
        axes[1].set_title('where')
        axes[1].axis('off')
        # full image
        img_path = os.path.join(picture_dir, info['file'])
        img = plt.imread(img_path)
        axes[2].imshow(img)
        axes[2].set_title('image')
        axes[2].axis('off')
        plt.show()
        
        print(f"{info['basename']}: agreement={similarity:.4f}")
        print(f"who norm: {np.linalg.norm(who_image_emb):.4f}, where norm: {np.linalg.norm(where_image_emb):.4f}")

        # 打印who和where的correlation
        if who_image_emb is not None and where_image_emb is not None and np.std(who_image_emb) > 0 and np.std(where_image_emb) > 0:
            who_where_corr = np.corrcoef(who_image_emb, where_image_emb)[0, 1]
            print(f"{info['basename']}: who-where correlation = {who_where_corr:.4f}")
        

# %% permutation test

# embedding algebra (king - queen = man - woman) stands for image as well
results = []

for info in image_info_list:
    image_emb = image_emb_dict.get(info['file'])
    if 'none' in info['file']:
        continue
    who_image_emb = image_emb_dict.get(info['who'] + '_none.png')
    #where_image_emb = image_emb_dict.get("none_" + info['where'] + '.png')
    random_where_key = "none_" + "desert" + '.png'
    where_image_emb = image_emb_dict.get(random_where_key)
    
    if image_emb is not None and who_image_emb is not None and where_image_emb is not None:
        algebra_emb = who_image_emb + where_image_emb
        if np.std(image_emb) > 0 and np.std(algebra_emb) > 0:
            similarity = np.corrcoef(image_emb, algebra_emb)[0, 1]
        else:
            similarity = float('nan')
        results.append({
            'image': info['basename'],
            'who': info['who'],
            'where': info['where'],
            'similarity': similarity
        })

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        # who image
        who_img_path = os.path.join(picture_dir, info['who'] + '_none.png')
        who_img = plt.imread(who_img_path)
        axes[0].imshow(who_img)
        axes[0].set_title('who')
        axes[0].axis('off')
        # where image
        #where_img_path = os.path.join(picture_dir, 'none_' + info['where'] + '.png')
        where_img_path = os.path.join(picture_dir, random_where_key)
        where_img = plt.imread(where_img_path)
        axes[1].imshow(where_img)
        axes[1].set_title('where')
        axes[1].axis('off')
        # full image
        img_path = os.path.join(picture_dir, info['file'])
        img = plt.imread(img_path)
        axes[2].imshow(img)
        axes[2].set_title('image')
        axes[2].axis('off')
        plt.show()
        
        print(f"{info['basename']}: agreement={similarity:.4f}")
        print(f"who norm: {np.linalg.norm(who_image_emb):.4f}, where norm: {np.linalg.norm(where_image_emb):.4f}")


# %%
# Create correlation matrix for all 'none' component images
none_images = []
none_embeddings = []

for name in picture_name_list:
    if 'none' in name:
        emb = image_emb_dict.get(name)
        if emb is not None:
            none_images.append(name)
            none_embeddings.append(emb)

# Sort: first xxx_none (who), then none_xxx (where)
who_items = [(name, emb) for name, emb in zip(none_images, none_embeddings) if name.endswith('_none.png')]
where_items = [(name, emb) for name, emb in zip(none_images, none_embeddings) if name.startswith('none_')]

# Remove none_none.png if it exists
none_none_items = [(name, emb) for name, emb in where_items if name == 'none_none.png']
if none_none_items:
    where_items.remove(none_none_items[0])

# Sort each group alphabetically
who_items.sort(key=lambda x: x[0])
where_items.sort(key=lambda x: x[0])

# Combine in order: who first, then where
sorted_items = who_items + where_items
none_images = [item[0] for item in sorted_items]
none_embeddings = np.array([item[1] for item in sorted_items])

# Convert to numpy array
# Compute correlation matrix
correlation_matrix = np.corrcoef(none_embeddings)

# Plot correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            xticklabels=none_images, 
            yticklabels=none_images,
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            annot=False,
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix of None Component Images')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(f"Total none component images: {len(none_images)}")

# %%
# Collect all images that don't contain 'none' in their filename
non_none_images = []
non_none_embeddings = []

for name in picture_name_list:
    if 'none' not in name.lower():
        emb = image_emb_dict.get(name)
        if emb is not None:
            non_none_images.append(name)
            non_none_embeddings.append(emb)

# Sort alphabetically
sorted_indices = np.argsort(non_none_images)
non_none_images = [non_none_images[i] for i in sorted_indices]
non_none_embeddings = np.array([non_none_embeddings[i] for i in sorted_indices])

# Compute correlation matrix
correlation_matrix_non_none = np.corrcoef(non_none_embeddings)

# Plot correlation matrix
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix_non_none, 
            xticklabels=non_none_images, 
            yticklabels=non_none_images,
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            annot=False,
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix of Non-None Images')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

print(f"Total non-none images: {len(non_none_images)}")

# print norm of each who (based on none images)
for who in who_list:
    who_emb = image_emb_dict.get(who + '_none.png')
    if who_emb is not None:
        print(f"{who}: {np.linalg.norm(who_emb):.4f}")

# %%
# For each person (who), compute average pairwise correlation across all places
# and plot against the norm of their xxx_none.png embedding

who_avg_correlations = {}
who_norms = {}

for who in who_list:
    # Get all images for this person (excluding none images)
    who_images = [name for name in non_none_images if name.startswith(who + '_')]
    
    if len(who_images) < 2:
        continue
    
    # Get embeddings for this person's images
    who_embeddings = np.array([image_emb_dict[name] for name in who_images])
    
    # Compute correlation matrix
    who_corr_matrix = np.corrcoef(who_embeddings)
    
    # Extract lower triangle (excluding diagonal)
    lower_triangle_indices = np.tril_indices_from(who_corr_matrix, k=-1)
    lower_triangle_values = who_corr_matrix[lower_triangle_indices]
    
    # Compute average correlation
    avg_correlation = np.mean(lower_triangle_values)
    who_avg_correlations[who] = avg_correlation
    
    # Get norm of xxx_none.png embedding
    none_name = who + '_none.png'
    none_emb = image_emb_dict.get(none_name)
    if none_emb is not None:
        who_norms[who] = np.linalg.norm(none_emb)
    
    print(f"{who}: avg_corr={avg_correlation:.4f}, norm={who_norms.get(who, 'N/A')}")

# Plot norm vs. average similarity
who_names = [who for who in who_list if who in who_avg_correlations and who in who_norms]
norms = [who_norms[who] for who in who_names]
avg_corrs = [who_avg_correlations[who] for who in who_names]

plt.figure(figsize=(10, 6))
plt.scatter(norms, avg_corrs, s=100, alpha=0.6)
for i, who in enumerate(who_names):
    plt.annotate(who, (norms[i], avg_corrs[i]), fontsize=9, alpha=0.7)
plt.xlabel('Norm of xxx_none.png Embedding')
plt.ylabel('Average Pairwise Correlation (across places)')
plt.title('Norm vs. Average Similarity for Each Person')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compute correlation between norm and average similarity
if len(norms) > 1:
    corr_coef = np.corrcoef(norms, avg_corrs)[0, 1]
    print(f"\nCorrelation between norm and average similarity: {corr_coef:.4f}")


# %%
# Collect all images that don't contain 'none' in their filename
non_none_images = []
non_none_embeddings = []

for name in picture_name_list:
    if 'none' not in name.lower():
        emb = image_emb_dict.get(name)
        if emb is not None:
            non_none_images.append(name)
            non_none_embeddings.append(emb)

# Filter out park2 and mall2
filtered_non_none_images = []
for name in non_none_images:
    if 'park2' not in name.lower() and 'mall2' not in name.lower():
        filtered_non_none_images.append(name)

non_none_images = filtered_non_none_images

# Group images by place (where)
place_groups = {}
for name in non_none_images:
    # Extract place from filename (assuming format: who_where.png)
    parts = name.replace('.png', '').split('_')
    if len(parts) >= 2:
        place = parts[1]
        if place not in place_groups:
            place_groups[place] = []
        place_groups[place].append(name)

# Sort images within each place group and concatenate
sorted_non_none_images = []
for place in sorted(place_groups.keys()):
    sorted_non_none_images.extend(sorted(place_groups[place]))

# Reorder embeddings according to the new sorting
name_to_emb = {name: image_emb_dict[name] for name in non_none_images}
non_none_embeddings = np.array([name_to_emb[name] for name in sorted_non_none_images])

# Compute correlation matrix
correlation_matrix_non_none = np.corrcoef(non_none_embeddings)

# Plot correlation matrix
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix_non_none, 
            xticklabels=sorted_non_none_images, 
            yticklabels=sorted_non_none_images,
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1,
            annot=False,
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix of Non-None Images (Grouped by Place)')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

print(f"Total non-none images: {len(sorted_non_none_images)}")

# print norm of each place (based on none images)
for place in sorted(place_groups.keys()):
    place_emb = image_emb_dict.get('none_' + place + '.png')
    if place_emb is not None:
        print(f"{place}: {np.linalg.norm(place_emb):.4f}")
# %%
# text replication


# %% 
# Gender-occupation word pairs for embedding algebra analysis
gender_occupation_pairs = [
    ('king', 'queen'),
    ('man', 'woman'),
    ('actor', 'actress'),
    ('waiter', 'waitress'),
    ('prince', 'princess'),
    ('husband', 'wife'),
    ('father', 'mother'),
    ('son', 'daughter'),
    ('brother', 'sister'),
    ('uncle', 'aunt'),
    ('nephew', 'niece'),
    ('grandfather', 'grandmother'),
    ('boy', 'girl'),
    ('gentleman', 'lady'),
    ('sir', 'madam'),
    ('hero', 'heroine'),
    ('host', 'hostess'),
    ('steward', 'stewardess'),
    ('policeman', 'policewoman'),
    ('businessman', 'businesswoman')
]

# Get text embeddings using the model
def get_text_embedding(text):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(text_tokens)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

# Collect embeddings for all words
word_embeddings = {}
for male_word, female_word in gender_occupation_pairs:
    word_embeddings[male_word] = get_text_embedding(male_word)
    word_embeddings[female_word] = get_text_embedding(female_word)

# Calculate cosine similarity between male and female versions for each pair
# and the norm of the occupation (using male version as reference)
results_gender_occupation = []

for male_word, female_word in gender_occupation_pairs:
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Cosine similarity between male and female
    cos_sim = np.dot(male_emb, female_emb) / (np.linalg.norm(male_emb) * np.linalg.norm(female_emb))
    
    # Norm of the occupation (male version)
    occupation_norm = np.linalg.norm(male_emb)
    
    results_gender_occupation.append({
        'male': male_word,
        'female': female_word,
        'cos_similarity': cos_sim,
        'occupation_norm': occupation_norm
    })
    
    print(f"{male_word} vs {female_word}: cos_sim={cos_sim:.4f}, norm={occupation_norm:.4f}")

# Plot correlation between cosine similarity and occupation norm
cos_sims = [r['cos_similarity'] for r in results_gender_occupation]
occupation_norms = [r['occupation_norm'] for r in results_gender_occupation]

#%%
plt.figure(figsize=(10, 6))
plt.scatter(occupation_norms, cos_sims, alpha=0.6, s=100)

# Add labels for each point
for r in results_gender_occupation:
    plt.annotate(f"{r['male']}/{r['female']}", 
                xy=(r['occupation_norm'], r['cos_similarity']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

# Calculate and display correlation
correlation = np.corrcoef(occupation_norms, cos_sims)[0, 1]
plt.title(f'Male vs Female Cosine Similarity vs Occupation Norm\nCorrelation: {correlation:.4f}')
plt.xlabel('Occupation Norm (Male Version)')
plt.ylabel('Male-Female Cosine Similarity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nCorrelation between occupation norm and male-female similarity: {correlation:.4f}")

# %%
# Compare norm of man_none.png image embedding with 'man' word embedding
man_image_emb = image_emb_dict.get('man_none.png')
man_word_emb = word_embeddings.get('man')

if man_image_emb is not None and man_word_emb is not None:
    man_image_norm = np.linalg.norm(man_image_emb)
    man_word_norm = np.linalg.norm(man_word_emb)
    
    print(f"\nComparison of 'man' embeddings:")
    print(f"man_none.png image embedding norm: {man_image_norm:.4f}")
    print(f"'man' word embedding norm: {man_word_norm:.4f}")
    print(f"Ratio (image/word): {man_image_norm/man_word_norm:.4f}")
else:
    if man_image_emb is None:
        print("Warning: man_none.png embedding not found")
    if man_word_emb is None:
        print("Warning: 'man' word embedding not found")

# all words are normlized

# %%

# as text embeddings are normalized, the norm of the word embedding is 1
# we can't use norm to compare the between-context similarity
# instead, we can use cosine similarity between mean

# Gender-occupation word pairs for embedding algebra analysis
gender_occupation_pairs = [
    ('king', 'queen'),
    ('man', 'woman'),
    ('actor', 'actress'),
    ('waiter', 'waitress'),
    ('prince', 'princess'),
    ('husband', 'wife'),
    ('father', 'mother'),
    ('son', 'daughter'),
    ('brother', 'sister'),
    ('uncle', 'aunt'),
    ('nephew', 'niece'),
    ('grandfather', 'grandmother'),
    ('boy', 'girl'),
    ('gentleman', 'lady'),
    ('sir', 'madam'),
    ('hero', 'heroine'),
    ('host', 'hostess'),
    ('steward', 'stewardess'),
    ('policeman', 'policewoman'),
    ('businessman', 'businesswoman')
]

# Get text embeddings using the model
def get_text_embedding(text):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(text_tokens)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

# Collect embeddings for all words
word_embeddings = {}
for male_word, female_word in gender_occupation_pairs:
    word_embeddings[male_word] = get_text_embedding(male_word)
    word_embeddings[female_word] = get_text_embedding(female_word)

# Calculate man + woman as the raw gender embedding
man_emb = word_embeddings['man']
woman_emb = word_embeddings['woman']
man_woman_sum = man_emb + woman_emb

# Calculate cosine similarity between male and female versions for each pair
# and the correlation to man+woman sum
results_gender_occupation = []

for male_word, female_word in gender_occupation_pairs:
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Cosine similarity between male and female
    cos_sim = np.dot(male_emb, female_emb) / (np.linalg.norm(male_emb) * np.linalg.norm(female_emb))
    
    # Sum of male and female embeddings
    occupation_sum = male_emb + female_emb
    
    # Correlation between occupation sum and man+woman sum
    if np.std(occupation_sum) > 0 and np.std(man_woman_sum) > 0:
        occupation_distance = np.corrcoef(occupation_sum, man_woman_sum)[0, 1]
    else:
        occupation_distance = float('nan')
    
    results_gender_occupation.append({
        'male': male_word,
        'female': female_word,
        'cos_similarity': cos_sim,
        'occupation_norm': occupation_distance
    })
    
    print(f"{male_word} vs {female_word}: cos_sim={cos_sim:.4f}, distance_to_raw={occupation_distance:.4f}")

# Plot correlation between cosine similarity and occupation distance to raw
cos_sims = [r['cos_similarity'] for r in results_gender_occupation]
occupation_distances = [r['occupation_norm'] for r in results_gender_occupation]

#%%
plt.figure(figsize=(10, 6))
plt.scatter(occupation_distances, cos_sims, alpha=0.6, s=100)

# Add labels for each point
for r in results_gender_occupation:
    plt.annotate(f"{r['male']}/{r['female']}", 
                xy=(r['occupation_norm'], r['cos_similarity']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

# Calculate and display correlation
correlation = np.corrcoef(occupation_norms, cos_sims)[0, 1]
plt.title(f'Male vs Female Cosine Similarity vs Occupation Norm\nCorrelation: {correlation:.4f}')
plt.xlabel('Occupation Norm (Male Version)')
plt.ylabel('Male-Female Cosine Similarity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nCorrelation between occupation norm and male-female similarity: {correlation:.4f}")

# %%
# Calculate embedding differences for each pair
embedding_diffs = []
pair_labels = []

for r in results_gender_occupation:
    male_word = r['male']
    female_word = r['female']
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Calculate the difference between male and female embeddings
    diff = male_emb - female_emb
    embedding_diffs.append(diff)
    pair_labels.append(f"{male_word}-{female_word}")

# Convert to numpy array for easier manipulation
embedding_diffs = np.array(embedding_diffs)

# Calculate correlation matrix between all pairs of embedding differences
n_pairs = len(embedding_diffs)
correlation_matrix = np.zeros((n_pairs, n_pairs))

for i in range(n_pairs):
    for j in range(n_pairs):
        if np.std(embedding_diffs[i]) > 0 and np.std(embedding_diffs[j]) > 0:
            correlation_matrix[i, j] = np.corrcoef(embedding_diffs[i], embedding_diffs[j])[0, 1]
        else:
            correlation_matrix[i, j] = float('nan')

# Plot heatmap
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(n_pairs), pair_labels, rotation=90, ha='right')
plt.yticks(range(n_pairs), pair_labels)
plt.title('Correlation Heatmap of Embedding Differences (Male - Female)')
plt.tight_layout()
plt.show()

print(f"\nCorrelation matrix shape: {correlation_matrix.shape}")
print(f"Mean correlation (excluding diagonal): {np.mean(correlation_matrix[~np.eye(n_pairs, dtype=bool)]):.4f}")

# %%
# Calculate differences between each occupation pair and man-woman
man_emb = word_embeddings['man']
woman_emb = word_embeddings['woman']

male_to_man_diffs = []
female_to_woman_diffs = []
pair_labels_new = []

for r in results_gender_occupation:
    male_word = r['male']
    female_word = r['female']
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Calculate differences
    male_to_man_diff = male_emb - man_emb
    female_to_woman_diff = female_emb - woman_emb
    
    male_to_man_diffs.append(male_to_man_diff)
    female_to_woman_diffs.append(female_to_woman_diff)
    pair_labels_new.append(f"{male_word}-{female_word}")

# Calculate correlations between male-man and female-woman differences for each pair
correlations = []
for i in range(len(male_to_man_diffs)):
    if np.std(male_to_man_diffs[i]) > 0 and np.std(female_to_woman_diffs[i]) > 0:
        corr = np.corrcoef(male_to_man_diffs[i], female_to_woman_diffs[i])[0, 1]
        correlations.append(corr)
    else:
        correlations.append(float('nan'))

# Plot the correlations
plt.figure(figsize=(14, 6))
x_positions = np.arange(len(pair_labels_new))

# Plot male-man correlations (pink) and female-woman correlations (blue)
# Since we're comparing each pair, we'll plot the correlation values
plt.scatter(x_positions, correlations, c='purple', s=100, alpha=0.6, label='Correlation (male-man vs female-woman)')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Occupation Pairs')
plt.ylabel('Correlation')
plt.title('Correlation between (Male - Man) and (Female - Woman) Embeddings')
plt.xticks(x_positions, pair_labels_new, rotation=90, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nMean correlation between (male-man) and (female-woman): {np.nanmean(correlations):.4f}")

# %%
# Calculate norms and similarities for each pair
male_norms = []
female_norms = []
pair_similarities = []

for i, r in enumerate(results_gender_occupation):
    male_word = r['male']
    female_word = r['female']
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Calculate norms of (male - man) and (female - woman)
    male_to_man_norm = np.linalg.norm(male_emb - man_emb)
    female_to_woman_norm = np.linalg.norm(female_emb - woman_emb)
    
    male_norms.append(male_to_man_norm)
    female_norms.append(female_to_woman_norm)
    
    # Calculate similarity between male and female words
    similarity = np.dot(male_emb, female_emb) / (np.linalg.norm(male_emb) * np.linalg.norm(female_emb))
    pair_similarities.append(similarity)

# Calculate average norms for male and female
mean_male_norm = np.mean(male_norms)
mean_female_norm = np.mean(female_norms)

print(f"\nMean norm of (male - man): {mean_male_norm:.4f}")
print(f"Mean norm of (female - woman): {mean_female_norm:.4f}")
print(f"\nPair-wise statistics:")
for i, label in enumerate(pair_labels_new):
    print(f"{label}: male_norm={male_norms[i]:.4f}, female_norm={female_norms[i]:.4f}, similarity={pair_similarities[i]:.4f}")

# %%
# Calculate correlation between F_word and (M_word - man + woman)
# For all female-male word pairs
predicted_female_embeddings = []
actual_female_embeddings = []

for i, r in enumerate(results_gender_occupation):
    male_word = r['male']
    female_word = r['female']
    male_emb = word_embeddings[male_word]
    female_emb = word_embeddings[female_word]
    
    # Calculate: M_word - man + woman
    predicted_female = male_emb - man_emb + woman_emb
    
    predicted_female_embeddings.append(predicted_female)
    actual_female_embeddings.append(female_emb)

# Calculate correlation for each dimension
predicted_female_matrix = np.array(predicted_female_embeddings)
actual_female_matrix = np.array(actual_female_embeddings)

# Flatten to calculate overall correlation
predicted_flat = predicted_female_matrix.flatten()
actual_flat = actual_female_matrix.flatten()

overall_correlation = np.corrcoef(predicted_flat, actual_flat)[0, 1]

print(f"\nCorrelation between F_word and (M_word - man + woman):")
print(f"Overall correlation: {overall_correlation:.4f}")

# Calculate correlation for each word pair
pair_correlations = []
for i in range(len(predicted_female_embeddings)):
    corr = np.corrcoef(predicted_female_embeddings[i], actual_female_embeddings[i])[0, 1]
    pair_correlations.append(corr)
    print(f"{pair_labels_new[i]}: {corr:.4f}")

print(f"\nMean pair-wise correlation: {np.mean(pair_correlations):.4f}")

# Calculate cosine similarities
cosine_similarities = []
for i in range(len(predicted_female_embeddings)):
    pred = predicted_female_embeddings[i]
    actual = actual_female_embeddings[i]
    cos_sim = np.dot(pred, actual) / (np.linalg.norm(pred) * np.linalg.norm(actual))
    cosine_similarities.append(cos_sim)

print(f"\nMean cosine similarity: {np.mean(cosine_similarities):.4f}")

# %%
# New analysis: man/woman in different contexts
# Analyze M (man at context), F (woman at context), and context itself
# Compare their similarities to base concepts (man, woman, home)


# Define base embeddings
man_emb = get_text_embedding('man at home')
woman_emb = get_text_embedding('woman at home')
home_emb = get_text_embedding('home')

# Define various contexts/locations
contexts = ['mall', 'office', 'kitchen', 'hospital', 'school', 'park', 'gym', 'library']

# Store results
context_labels = []
m_to_man_sims = []
f_to_woman_sims = []
context_to_home_sims = []
f_minus_m_sims = []

for context in contexts:
    # Try to get embeddings for "man at context" and "woman at context"
    # We'll use simple concatenation or try different phrasings
    m_phrase = f"man at {context}"
    f_phrase = f"woman at {context}"
    
    # Try to get embeddings (may need to use context word itself as proxy)
    # For simplicity, we'll use the context word embedding
    context_emb = get_text_embedding(context)
    
    # Approximate M and F as man/woman + context
    m_emb = man_emb + context_emb
    f_emb = woman_emb + context_emb
    
    # Calculate similarities
    m_to_man = np.dot(m_emb, man_emb) / (np.linalg.norm(m_emb) * np.linalg.norm(man_emb))
    f_to_woman = np.dot(f_emb, woman_emb) / (np.linalg.norm(f_emb) * np.linalg.norm(woman_emb))
    context_to_home = np.dot(context_emb, home_emb) / (np.linalg.norm(context_emb) * np.linalg.norm(home_emb))
    
    # Calculate F-M difference (as cosine distance)
    f_minus_m = f_emb - m_emb
    f_minus_m_sim = np.linalg.norm(f_minus_m)
    
    context_labels.append(context)
    m_to_man_sims.append(m_to_man)
    f_to_woman_sims.append(f_to_woman)
    context_to_home_sims.append(context_to_home)
    f_minus_m_sims.append(f_minus_m_sim)

# Plot 1: Similarities across contexts
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(context_labels))
width = 0.25

plt.bar(x_pos - width, m_to_man_sims, width, label='M to man', alpha=0.8)
plt.bar(x_pos, f_to_woman_sims, width, label='F to woman', alpha=0.8)
plt.bar(x_pos + width, context_to_home_sims, width, label='Context to home', alpha=0.8)

plt.xlabel('Context')
plt.ylabel('Cosine Similarity')
plt.title('Similarities of M, F, and Context to Base Concepts')
plt.xticks(x_pos, context_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('context_similarities.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Context-to-home similarity vs F-M difference
plt.figure(figsize=(10, 6))
plt.scatter(context_to_home_sims, f_minus_m_sims, s=100, alpha=0.6)

# Add labels for each point
for i, label in enumerate(context_labels):
    plt.annotate(label, (context_to_home_sims[i], f_minus_m_sims[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Context-to-Home Similarity')
plt.ylabel('F-M Difference (L2 norm)')
plt.title('Relationship between Context-Home Similarity and Gender Difference')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('context_home_vs_gender_diff.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nContext Analysis Results:")
print("-" * 60)
for i, context in enumerate(context_labels):
    print(f"{context:12s} | M→man: {m_to_man_sims[i]:.4f} | F→woman: {f_to_woman_sims[i]:.4f} | "
          f"Context→home: {context_to_home_sims[i]:.4f} | F-M diff: {f_minus_m_sims[i]:.4f}")

# %%
gender_occupation_pairs = [
    ('man', 'woman'),
    ('king', 'queen'),
    ('actor', 'actress'),
    ('waiter', 'waitress'),
    ('prince', 'princess'),
    ('husband', 'wife'),
    ('father', 'mother'),
    ('son', 'daughter'),
    ('brother', 'sister'),
    ('uncle', 'aunt'),
    ('nephew', 'niece'),
    ('grandfather', 'grandmother'),
    ('boy', 'girl'),
    ('gentleman', 'lady'),
    ('sir', 'madam'),
    ('hero', 'heroine'),
    ('host', 'hostess'),
    ('steward', 'stewardess'),
    ('policeman', 'policewoman'),
    ('businessman', 'businesswoman')
]

# Get embeddings for each word pair (not normalized)
avg_norms = []
mf_cosine_sims = []
pair_labels = []

for male_word, female_word in gender_occupation_pairs:
    # Get embeddings without normalization
    m_emb = get_text_embedding(male_word)
    f_emb = get_text_embedding(female_word)
    
    # Calculate average embedding and its norm
    avg_emb = (m_emb + f_emb) / 2
    avg_norm = np.linalg.norm(avg_emb)
    
    # Calculate cosine similarity between M and F
    mf_cosine_sim = np.dot(m_emb, f_emb) / (np.linalg.norm(m_emb) * np.linalg.norm(f_emb))
    
    avg_norms.append(avg_norm)
    mf_cosine_sims.append(mf_cosine_sim)
    pair_labels.append(f"{male_word}-{female_word}")

# Calculate correlation coefficient
correlation = np.corrcoef(avg_norms, mf_cosine_sims)[0, 1]

# Plot: Average norm vs M-F cosine similarity
plt.figure(figsize=(12, 8))
plt.scatter(avg_norms, mf_cosine_sims, s=100, alpha=0.6)

# Add labels for each point
for i, label in enumerate(pair_labels):
    plt.annotate(label, (avg_norms[i], mf_cosine_sims[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('L2 Norm of Average Embedding ||(F+M)/2||')
plt.ylabel('Cosine Similarity between M and F')
plt.title(f'Gender Word Pairs: Average Embedding Norm vs M-F Cosine Similarity\nCorrelation: {correlation:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gender_pairs_avg_vs_cosine.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGender Pair Analysis:")
print("-" * 70)
for i, label in enumerate(pair_labels):
    print(f"{label:25s} | Avg norm: {avg_norms[i]:.4f} | M-F cosine sim: {mf_cosine_sims[i]:.4f}")
print(f"\nCorrelation (avg norm vs M-F cosine sim): {correlation:.4f}")


# %% Implementation using word2vec
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

# Load the word2vec model
model = api.load('word2vec-google-news-300')
word_vectors = model.wv

# Calculate the average embedding norm and M-F cosine similarity for each gender occupation pair using word2vec
avg_norms_w2v = []
mf_cosine_sims_w2v = []
pair_labels_w2v = []

for male_word, female_word in gender_occupation_pairs:
    # Check if both words are in the word2vec vocabulary
    if male_word in word_vectors and female_word in word_vectors:
        m_vec = word_vectors[male_word]
        f_vec = word_vectors[female_word]
        
        avg_vec = (m_vec + f_vec) / 2.0
        avg_norm = np.linalg.norm(avg_vec)
        cosine_sim = np.dot(m_vec, f_vec) / (np.linalg.norm(m_vec) * np.linalg.norm(f_vec))
        
        avg_norms_w2v.append(avg_norm)
        mf_cosine_sims_w2v.append(cosine_sim)
        pair_labels_w2v.append(f"{male_word}-{female_word}")

# Compute the correlation coefficient
if len(avg_norms_w2v) > 1:
    correlation_w2v = np.corrcoef(avg_norms_w2v, mf_cosine_sims_w2v)[0, 1]
else:
    correlation_w2v = float('nan')

# Plot scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(avg_norms_w2v, mf_cosine_sims_w2v, s=100, alpha=0.6)
for i, label in enumerate(pair_labels_w2v):
    plt.annotate(label, (avg_norms_w2v[i], mf_cosine_sims_w2v[i]), xytext=(5,5), textcoords='offset points', fontsize=8)

plt.xlabel('L2 Norm: (F+M)/2 (word2vec)')
plt.ylabel('Cosine Similarity (word2vec)')
plt.title(f'Gender Occupation Word Pairs (word2vec): Average Embedding Norm vs Cosine Similarity\nCorrelation: {correlation_w2v:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gender_pairs_avg_vs_cosine_word2vec.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGender Pair Analysis (word2vec):")
print("-" * 70)
for i, label in enumerate(pair_labels_w2v):
    print(f"{label:25s} | Avg norm: {avg_norms_w2v[i]:.4f} | M-F cosine sim: {mf_cosine_sims_w2v[i]:.4f}")
print(f"\nCorrelation (avg norm vs M-F cosine sim): {correlation_w2v:.4f}")


# %% helper function
def analyze_gender_pairs(get_embedding_func, gender_pairs, model_name=""):
    """
    Analyze gender word pairs by plotting average embedding norm vs M-F cosine similarity.
    
    Args:
        get_embedding_func: Function that takes a word and returns its embedding
        gender_pairs: List of (male_word, female_word) tuples
        model_name: Name of the model for plot titles and filenames
    """
    avg_norms = []
    mf_cosine_sims = []
    pair_labels = []
    
    for male_word, female_word in gender_pairs:
        try:
            # Get embeddings without normalization
            m_emb = get_embedding_func(male_word)
            f_emb = get_embedding_func(female_word)
            
            # Calculate average embedding and its norm
            avg_emb = (m_emb + f_emb) / 2
            avg_norm = np.linalg.norm(avg_emb)
            
            # Calculate cosine similarity between M and F
            mf_cosine_sim = np.dot(m_emb, f_emb) / (np.linalg.norm(m_emb) * np.linalg.norm(f_emb))
            
            avg_norms.append(avg_norm)
            mf_cosine_sims.append(mf_cosine_sim)
            pair_labels.append(f"{male_word}-{female_word}")
        except KeyError:
            print(f"Warning: Skipping pair ({male_word}, {female_word}) - word not found in vocabulary")
            continue
    
    if len(avg_norms) == 0:
        print("No valid word pairs found!")
        return
    
    # Calculate correlation coefficient
    if len(avg_norms) > 1:
        correlation = np.corrcoef(avg_norms, mf_cosine_sims)[0, 1]
    else:
        correlation = float('nan')
    
    # Plot: Average norm vs M-F cosine similarity
    plt.figure(figsize=(12, 8))
    plt.scatter(avg_norms, mf_cosine_sims, s=100, alpha=0.6)
    
    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (avg_norms[i], mf_cosine_sims[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('L2 Norm of Average Embedding ||(F+M)/2||')
    plt.ylabel('Cosine Similarity between M and F')
    
    title_suffix = f" ({model_name})" if model_name else ""
    plt.title(f'Gender Word Pairs: Average Embedding Norm vs M-F Cosine Similarity{title_suffix}\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename_suffix = f"_{model_name}" if model_name else ""
    plt.savefig(f'gender_pairs_avg_vs_cosine{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGender Pair Analysis{title_suffix}:")
    print("-" * 70)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Avg norm: {avg_norms[i]:.4f} | M-F cosine sim: {mf_cosine_sims[i]:.4f}")
    print(f"\nCorrelation (avg norm vs M-F cosine sim): {correlation:.4f}")
    
    #return avg_norms, mf_cosine_sims, pair_labels, correlation


