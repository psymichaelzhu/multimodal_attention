# %% load clip model and tokenizer
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

def get_clip_embedding(text):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(text_tokens)
        return text_features.cpu().detach().numpy().flatten()

# %% helper function
def analyze_gender_pairs(get_embedding_func, gender_pairs, model_name=""):
    """
    Analyze gender word pairs by plotting custom norm (see below) vs M-F cosine similarity.

    For each pair (M, F): x-axis is the norm of the average of (F-woman) and (M-man),
    y-axis is cosine similarity between M and F.
    """
    custom_norms = []
    mf_cosine_sims = []
    pair_labels = []

    # Get embeddings for 'man' and 'woman' for reference
    try:
        man_emb = get_embedding_func('man')
        woman_emb = get_embedding_func('woman')
    except KeyError:
        print("Warning: 'man' or 'woman' not in vocabulary. Aborting analysis.")
        return

    for male_word, female_word in gender_pairs:
        try:
            m_emb = get_embedding_func(male_word)
            f_emb = get_embedding_func(female_word)

            # Custom: norm of average of (F-woman) and (M-man)
            m_minus_man = m_emb - man_emb
            f_minus_woman = f_emb - woman_emb
            avg_vec = (m_minus_man + f_minus_woman) / 2
            custom_norm = np.linalg.norm(avg_vec)

            # Cosine similarity between M and F
            mf_cosine_sim = np.dot(m_emb, f_emb) / (np.linalg.norm(m_emb) * np.linalg.norm(f_emb))

            custom_norms.append(custom_norm)
            mf_cosine_sims.append(mf_cosine_sim)
            pair_labels.append(f"{male_word}-{female_word}")
        except KeyError:
            print(f"Warning: Skipping pair ({male_word}, {female_word}) - word not found in vocabulary")
            continue

    if len(custom_norms) == 0:
        print("No valid word pairs found!")
        return

    # Correlation coefficient
    if len(custom_norms) > 1:
        correlation = np.corrcoef(custom_norms, mf_cosine_sims)[0,1]
    else:
        correlation = float('nan')

    # Plot: Custom norm vs M-F cosine similarity
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, mf_cosine_sims, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], mf_cosine_sims[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel('Norm of average: (F-woman + M-man)/2')
    plt.ylabel('Cosine Similarity between M and F')

    title_suffix = f" ({model_name})" if model_name else ""
    plt.title(f'Gender Word Pairs: Custom Norm vs M-F Cosine Similarity{title_suffix}\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename_suffix = f"_{model_name}" if model_name else ""
    plt.savefig(f'gender_pairs_customnorm_vs_cosine{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGender Pair Analysis{title_suffix}:")
    print("-" * 70)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Custom norm: {custom_norms[i]:.4f} | M-F cosine sim: {mf_cosine_sims[i]:.4f}")
    print(f"\nCorrelation (custom norm vs M-F cosine sim): {correlation:.4f}")

    #return custom_norms, mf_cosine_sims, pair_labels, correlation

#%% load data
import numpy as np
import matplotlib.pyplot as plt

# Define the gender occupation pairs
gender_occupation_pairs = [
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

# Define context-based gender occupation pairs: man at xxx, woman at xxx, where xxx is an environment
gender_occupation_pairs = [
    ('man at restaurant', 'woman at restaurant'),
    ('man at hospital', 'woman at hospital'),
    ('man at office', 'woman at office'),
    ('man at school', 'woman at school'),
    ('man at park', 'woman at park'),
    ('man at gym', 'woman at gym'),
    ('man at airport', 'woman at airport'),
    ('man at library', 'woman at library'),
    ('man at supermarket', 'woman at supermarket'),
    ('man at concert', 'woman at concert'),
    ('man at hotel', 'woman at hotel'),
    ('man at museum', 'woman at museum'),
    ('man at bank', 'woman at bank'),
    ('man at church', 'woman at church'),
    ('man at cinema', 'woman at cinema'),
    ('man at stadium', 'woman at stadium'),
    ('man at bar', 'woman at bar'),
    ('man at club', 'woman at club'),
    ('man at university', 'woman at university')
]

# Example usage with different models
#analyze_gender_pairs(lambda x: wv_model[x], gender_occupation_pairs, "word2vec")
#analyze_gender_pairs(get_clip_embedding, gender_occupation_pairs, "clip")

# %% load word2vec model
import gensim.downloader as api
word2vec_model = api.load('word2vec-google-news-300')

def get_word2vec_embedding(word):
    return word2vec_model[word]

analyze_gender_pairs(get_word2vec_embedding, gender_occupation_pairs, "word2vec")


# %% test: analyze pair similarity
def analyze_pair_similarity(target1, target2):
    """
    Step 1: Output cosine similarity between (target1-target2) and (man-woman)
    Step 2: Output cosine similarity between target1 and target2
    """
    # Get vectors
    vec1 = get_word2vec_embedding(target1)
    vec2 = get_word2vec_embedding(target2)
    man_vec = get_word2vec_embedding('man')
    woman_vec = get_word2vec_embedding('woman')

    # Compute differences
    pair_diff = vec1 - vec2
    man_woman_diff = man_vec - woman_vec

    # Define cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Step 1: Cosine similarity between diffs
    diff_similarity = cosine_similarity(pair_diff, man_woman_diff)
    print(f"Step 1: Cosine similarity between '{target1}-{target2}' and 'man-woman': {diff_similarity:.4f}")

    # Step 2: Cosine similarity between individual words
    word_similarity = cosine_similarity(vec1, vec2)
    print(f"Step 2: Cosine similarity between '{target1}' and '{target2}': {word_similarity:.4f}")

# Example usage:
analyze_pair_similarity('king', 'queen')
analyze_pair_similarity('host', 'hostess')

# %% test classic

def find_analogy(wordA, wordB, wordC, topn=4):
    va = get_word2vec_embedding(wordA)
    vb = get_word2vec_embedding(wordB)
    vc = get_word2vec_embedding(wordC)
    v = va - vb + vc
    result = word2vec_model.most_similar(positive=[wordA, wordC], negative=[wordB], topn=topn+3)
    filtered_result = [(w, s) for (w, s) in result if w not in [wordA, wordB, wordC]][:topn]
    for i, (word, score) in enumerate(filtered_result, 1):
        print(f"{i}. {word:15s} similarity={score:.4f}")

find_analogy('man', 'woman', 'bride', topn=4)

# %% word2vec: gender occupation pairs
analyze_gender_pairs(get_word2vec_embedding, gender_occupation_pairs, "word2vec")

# %% clip: gender occupation pairs
analyze_gender_pairs(get_clip_embedding, gender_occupation_pairs, "clip")
# %% clustering: clip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Prepare embeddings for all words in gender_occupation_pairs (flattened)
all_words = set()
for w1, w2 in gender_occupation_pairs:
    all_words.add(w1)
    all_words.add(w2)
all_words = sorted(all_words)

embeddings = np.array([get_clip_embedding(w) for w in all_words])

# First reduce dimensionality of embeddings using PCA
pca_cluster = PCA(n_components=16)  # You can adjust n_components as needed
embeddings_pca = pca_cluster.fit_transform(embeddings)

# Select optimal number of clusters via silhouette score
range_n_clusters = range(2, min(11, len(all_words)))
best_n_clusters = 2
best_score = -1
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings_pca)
    score = silhouette_score(embeddings_pca, cluster_labels)
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

# Fit final model
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0, n_init='auto')
labels = kmeans.fit_predict(embeddings)

# Print clusters and their members
clusters = {i: [] for i in range(best_n_clusters)}
for word, label in zip(all_words, labels):
    clusters[label].append(word)

print("Word clusters based on CLIP embeddings:")
for i in range(best_n_clusters):
    print(f"\nCluster {i+1}: {', '.join(clusters[i])}")

# 2D PCA and scatter plot
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
colors = plt.cm.get_cmap('tab10', best_n_clusters)

for i in range(best_n_clusters):
    idxs = np.where(labels == i)[0]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], 
                label=f'Cluster {i+1}', s=60, color=colors(i))

for i, word in enumerate(all_words):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=9, alpha=0.7)

plt.title('CLIP Embeddings of Words (PCA 2D Clustering)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.tight_layout()
plt.show()

# %% helper function
def analyze_gender_pairs(get_embedding_func, gender_pairs, model_name=""):
    """
    Analyze gender word pairs by plotting custom norm (see below) vs M-F cosine similarity.

    For each pair (M, F): x-axis is the norm of the average of (F-woman) and (M-man),
    y-axis is cosine similarity between M and F.
    """
    custom_norms = []
    mf_cosine_sims = []
    pair_labels = []

    # Get embeddings for 'man' and 'woman' for reference
    try:
        man_emb = get_embedding_func('man')
        woman_emb = get_embedding_func('woman')
    except KeyError:
        print("Warning: 'man' or 'woman' not in vocabulary. Aborting analysis.")
        return

    for male_word, female_word in gender_pairs:
        try:
            m_emb = get_embedding_func(male_word)
            f_emb = get_embedding_func(female_word)
            
            # Extract context from phrases like "woman at office" -> "office"
            if ' at ' in male_word:
                context_text = male_word.split(' at ')[1]

            context_emb = get_embedding_func(f'person at {context_text}')
            custom_norm = np.linalg.norm(context_emb)

            # Cosine similarity between M and F
            mf_cosine_sim = np.dot(m_emb, f_emb) / (np.linalg.norm(m_emb) * np.linalg.norm(f_emb))

            custom_norms.append(custom_norm)
            mf_cosine_sims.append(mf_cosine_sim)
            pair_labels.append(f"{male_word}-{female_word}")
        except KeyError:
            print(f"Warning: Skipping pair ({male_word}, {female_word}) - word not found in vocabulary")
            continue

    if len(custom_norms) == 0:
        print("No valid word pairs found!")
        return

    # Correlation coefficient
    if len(custom_norms) > 1:
        correlation = np.corrcoef(custom_norms, mf_cosine_sims)[0,1]
    else:
        correlation = float('nan')

    # Plot: Custom norm vs M-F cosine similarity
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, mf_cosine_sims, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], mf_cosine_sims[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel('Norm of average: (F-woman + M-man)/2')
    plt.ylabel('Cosine Similarity between M and F')

    title_suffix = f" ({model_name})" if model_name else ""
    plt.title(f'Gender Word Pairs: Custom Norm vs M-F Cosine Similarity{title_suffix}\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename_suffix = f"_{model_name}" if model_name else ""
    plt.savefig(f'gender_pairs_customnorm_vs_cosine{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGender Pair Analysis{title_suffix}:")
    print("-" * 70)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Custom norm: {custom_norms[i]:.4f} | M-F cosine sim: {mf_cosine_sims[i]:.4f}")
    print(f"\nCorrelation (custom norm vs M-F cosine sim): {correlation:.4f}")

    #return custom_norms, mf_cosine_sims, pair_labels, correlation


# %%
analyze_gender_pairs(get_clip_embedding, gender_occupation_pairs, "clip")

# word2vec won't apply, as it's not sentence-based
# %%
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


# %%
# Input text
text = "man at office"

# Tokenize and encode text using batch_encode_plus
# The function returns a dictionary containing the token IDs and attention masks
encoding = tokenizer.batch_encode_plus(
    [text],  # List of input texts
    padding=True,              # Pad to the maximum sequence length
    truncation=True,           # Truncate to the maximum sequence length if necessary
    return_tensors='pt',      # Return PyTorch tensors
    add_special_tokens=True    # Add special tokens CLS and SEP
)

input_ids = encoding['input_ids']  # Token IDs
# print input IDs
print(f"Input ID: {input_ids}")
attention_mask = encoding['attention_mask']  # Attention mask
# print attention mask
print(f"Attention mask: {attention_mask}")

# Generate embeddings using BERT model
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    # Extract sentence embedding using [CLS] token (first token)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

# Output the shape of sentence embedding
print(f"Shape of Sentence Embedding: {sentence_embedding.shape}")
print(f"Sentence embedding for '{text}': {sentence_embedding}")


# %% load sentence-transformer model
from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer("all-MiniLM-L6-v2")  # multi-language model

# %%
analyze_gender_pairs(st_model.encode, gender_occupation_pairs, "sentence-transformer")
# %%
