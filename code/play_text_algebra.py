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



#%% load data
import numpy as np
import matplotlib.pyplot as plt

# Define the gender occupation pairs
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

# %%
analyze_gender_pairs(get_word2vec_embedding, gender_occupation_pairs, "word2vec")

# %%
