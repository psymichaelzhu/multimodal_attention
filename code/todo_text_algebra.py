# %% preparation

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %% attempt: Find context words for each semantic gender pair

if False:
    # load word2vec model
    import gensim.downloader as api
    word2vec_model = api.load("word2vec-google-news-300")
    # test word2vec
    print(word2vec_model.most_similar("man"))
    def find_analogy(wordA, wordB, wordC, topn=4):
        va = word2vec_model[wordA]
        vb = word2vec_model[wordB]
        vc = word2vec_model[wordC]
        v = va - vb + vc
        result = word2vec_model.most_similar(positive=[wordA, wordC], negative=[wordB], topn=topn+3)
        filtered_result = [(w, s) for (w, s) in result if w not in [wordA, wordB, wordC]][:topn]
        for i, (word, score) in enumerate(filtered_result, 1):
            print(f"{i}. {word:15s} similarity={score:.4f}")

    find_analogy('man', 'woman', 'bride', topn=4)

    # not so good

    def find_context(list_pos, list_neg, topn=4):
        return word2vec_model.most_similar(positive=list_pos, negative=list_neg, topn=topn+3)

    print("\nFinding context words for semantic gender pairs:")
    print("=" * 80)

    for male_word, female_word in gender_context_pairs_semantic:
        print(f"\nPair: {male_word} - {female_word}")
        print("-" * 40)
        context_words = find_context([male_word, female_word], ['man', 'woman'], topn=5)
        for i, (word, score) in enumerate(context_words, 1):
            if word not in [male_word, female_word, 'man', 'woman']:
                print(f"{i}. {word:20s} similarity={score:.4f}")





# %% load clip model and tokenizer
import torch
from PIL import Image

# model info
model_source, model_name, pretrained_name = ('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b')
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




# %% helper function: clustering

def cluster_and_visualize_texts(pair_list, get_embedding_func, model_name=""):
    """
    Cluster all texts from a pair list and visualize them in 2D using PCA.
    
    Args:
        pair_list: List of tuples containing text pairs
        get_embedding_func: Function to get embeddings for text
        model_name: Name of the model for plot title
    
    Returns:
        clusters: Dictionary mapping cluster labels to lists of texts
        labels: Cluster labels for each text
    """
    # Collect all unique texts from the pair list
    all_texts = set()
    for text1, text2 in pair_list:
        all_texts.add(text1)
        all_texts.add(text2)
    all_texts = sorted(all_texts)
    
    # Get embeddings for all texts
    embeddings = np.array([get_embedding_func(text) for text in all_texts])
    # First reduce dimensionality of embeddings using PCA for clustering
    # Select components that explain 95% of variance
    pca_cluster = PCA(n_components=0.95, svd_solver='full')
    embeddings_pca = pca_cluster.fit_transform(embeddings)
    
    # Select optimal number of clusters via silhouette score
    range_n_clusters = range(2, min(11, len(all_texts)))
    best_n_clusters = 2
    best_score = -1
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings_pca)
        score = silhouette_score(embeddings_pca, cluster_labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    # Fit final model with optimal number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(embeddings_pca)
    
    # Organize texts into clusters
    clusters = {i: [] for i in range(best_n_clusters)}
    for text, label in zip(all_texts, labels):
        clusters[label].append(text)
    
    # Print clusters and their members
    print(f"\nText clusters based on {model_name} embeddings:")
    print(f"Optimal number of clusters: {best_n_clusters} (silhouette score: {best_score:.3f})")
    for i in range(best_n_clusters):
        print(f"\nCluster {i+1}: {', '.join(clusters[i])}")
    
    # 2D PCA for visualization
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', best_n_clusters)
    
    for i in range(best_n_clusters):
        idxs = np.where(labels == i)[0]
        plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], 
                    label=f'Cluster {i+1}', s=60, color=colors(i), alpha=0.7)
    
    # Add text labels for each point
    for i, text in enumerate(all_texts):
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], text, 
                fontsize=8, alpha=0.7, ha='center', va='bottom')
    
    title = f'{model_name} Embeddings Clustering (PCA 2D)' if model_name else 'Text Embeddings Clustering (PCA 2D)'
    plt.title(title)
    plt.xlabel(f'PCA 1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return clusters, labels

# %% helper function: test norm-shrink hypothesis
def analyze_word_pairs(get_embedding_func, anchor_pair, word_pairs, model_name=""):
    """
    Analyze word pairs by plotting custom norm (see below) vs cosine similarity between pairs.

    For each pair (word1, word2): x-axis is the norm of the average of (word2-anchor2) and (word1-anchor1),
    y-axis is cosine similarity between word1 and word2.
    
    Args:
        get_embedding_func: Function to get embeddings for words
        anchor_pair: Tuple of (anchor1, anchor2) - the reference pair (e.g., ("man", "woman"))
        word_pairs: List of (word1, word2) tuples to analyze
        model_name: Name of the model for labeling
    """
    custom_norms = []
    pair_cosine_sims = []
    pair_labels = []

    # Get embeddings for anchor words
    anchor1, anchor2 = anchor_pair
    try:
        anchor1_emb = get_embedding_func(anchor1)
        anchor2_emb = get_embedding_func(anchor2)
    except KeyError:
        print(f"Warning: {anchor1} or {anchor2} not in vocabulary. Aborting analysis.")
        return

    for word1, word2 in word_pairs:
        try:
            w1_emb = get_embedding_func(word1)
            w2_emb = get_embedding_func(word2)
            
            # Extract context from phrases like "woman at office" -> "office"
            if ' at ' in word1:
                context_text = word1.split(' at ')[1]

                context_emb = get_embedding_func(f'person at {context_text}')
                custom_norm = np.linalg.norm(context_emb)
                x_title = 'Norm of context'
            else:
                # norm of average of (word2-anchor2) and (word1-anchor1)
                w1_minus_anchor = w1_emb - anchor1_emb
                w2_minus_anchor = w2_emb - anchor2_emb
                avg_vec = (w1_minus_anchor + w2_minus_anchor) / 2
                custom_norm = np.linalg.norm(avg_vec)
                x_title = f'Norm of average: (word2-{anchor2} + word1-{anchor1})/2'

            # Cosine similarity between word1 and word2
            pair_cosine_sim = np.dot(w1_emb, w2_emb) / (np.linalg.norm(w1_emb) * np.linalg.norm(w2_emb))

            custom_norms.append(custom_norm)
            pair_cosine_sims.append(pair_cosine_sim)
            pair_labels.append(f"{word1}-{word2}")
        except KeyError:
            print(f"Warning: Skipping pair ({word1}, {word2}) - word not found in vocabulary")
            continue

    if len(custom_norms) == 0:
        print("No valid word pairs found!")
        return

    # Correlation coefficient
    if len(custom_norms) > 1:
        correlation = np.corrcoef(custom_norms, pair_cosine_sims)[0,1]
    else:
        correlation = float('nan')

    # Plot: Custom norm vs pair cosine similarity
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, pair_cosine_sims, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], pair_cosine_sims[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel(x_title)
    plt.ylabel(f'Cosine Similarity between word pairs')

    title_suffix = f" ({model_name})" if model_name else ""
    plt.title(f'Word Pairs (anchor: {anchor1}-{anchor2}): Custom Norm vs Cosine Similarity{title_suffix}\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename_suffix = f"_{model_name}" if model_name else ""
    plt.savefig(f'word_pairs_customnorm_vs_cosine{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nWord Pair Analysis{title_suffix} (anchor: {anchor1}-{anchor2}):")
    print("-" * 70)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Custom norm: {custom_norms[i]:.4f} | Pair cosine sim: {pair_cosine_sims[i]:.4f}")
    print(f"\nCorrelation (custom norm vs pair cosine sim): {correlation:.4f}")

    #return custom_norms, pair_cosine_sims, pair_labels, correlation


# %% run: gender pairs

# gender + semantic context, one word
gender_context_pairs_semantic = [
    ('king', 'queen'),# royal
    ('actor', 'actress'),# acting
    ('waiter', 'waitress'),# serving
    ('prince', 'princess'),# royal
    ('husband', 'wife'),# married
    ('father', 'mother'),# parent
    ('son', 'daughter'),# child
    ('brother', 'sister'),# sibling
    ('uncle', 'aunt'),# relative
    ('nephew', 'niece'),# relative
    ('grandfather', 'grandmother'),# grandparent
    ('boy', 'girl'),# adolescent
    ('gentleman', 'lady'),# respectable
    ('sir', 'madam'),# respectable
    ('policeman', 'policewoman'), # police
    ('businessman', 'businesswoman'), # business
]
analyze_word_pairs(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic, "clip")
# reverse testing
analyze_word_pairs(get_clip_embedding, ("female", "male"), gender_context_pairs_semantic, "clip")


# %% run: man/woman location pairs


# gender + location, a phrase
gender_context_pairs_combinational = [
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
analyze_word_pairs(get_clip_embedding, ("man", "woman"), gender_context_pairs_combinational, "clip")
# %% run: clustering gender pairs
cluster_and_visualize_texts(gender_context_pairs_semantic, get_clip_embedding, "clip")
cluster_and_visualize_texts(gender_context_pairs_combinational, get_clip_embedding, "clip")


# %% run: adult/young pairs
adult_young_pairs = [
    # extremely common
    ('cat', 'kitten'),
    ('dog', 'puppy'),
    ('cow', 'calf'),
    ('sheep', 'lamb'),
    ('horse', 'foal'),
    ('pig', 'piglet'),
    ('chicken', 'chick'),
    ('duck', 'duckling'),
    ('goose', 'gosling'),
    ('deer', 'fawn'),
    ('rabbit', 'bunny'),
    ('bear', 'cub'),
    ('lion', 'cub'),
    ('tiger', 'cub'),
    ('leopard', 'cub'),
    ('cheetah', 'cub'),
    ('fox', 'kit'),

    # medium common
    ('owl', 'owlet'),
    ('swan', 'cygnet'),
    ('hawk', 'eyas'),
    ('eagle', 'eaglet'),
    ('turkey', 'poult'),
    ('pigeon', 'squab'),
    ('fish', 'fry'),
    ('frog', 'tadpole'),

    # professional but common
    ('insect', 'larva'),
    ('butterfly', 'caterpillar'),
    ('moth', 'caterpillar'),
    ('salmon', 'smolt')
]

analyze_word_pairs(get_clip_embedding, ("adult", "young"), adult_young_pairs, "clip")
# reverse testing
analyze_word_pairs(get_clip_embedding, ("young", "adult"), adult_young_pairs, "clip")
# %% run: singular/collective pairs
singular_collective_pairs = [
    ('fish', 'school'),
    ('cow', 'cattle'),
    ('sheep', 'flock'),
    ('bird', 'flock'),
    ('wolf', 'pack'),
    ('dog', 'pack'),
    ('bee', 'swarm'),
    ('ant', 'colony'),
    ('goose', 'gaggle'),
    ('duck', 'raft'),
    ('horse', 'herd'),
    ('pig', 'herd')
]
analyze_word_pairs(get_clip_embedding, ("singular", "collective"), singular_collective_pairs, "clip")
# %% run: ingredient/product pairs
transformation_pairs = [
    ('grape', 'raisin'),
    ('plum', 'prune'),
    ('corn', 'popcorn'),
    ('grain', 'flour'),
    ('wheat', 'flour'),
    ('wood', 'lumber'),
    ('tree', 'timber'),
    ('cacao', 'chocolate'),
    ('milk', 'cheese'),
    ('cream', 'butter'),
    ('clay', 'ceramic'),
    ('sand', 'glass')
]
analyze_word_pairs(get_clip_embedding, ("ingredient", "product"), transformation_pairs, "clip")
# %% run: leader/follower pairs
leader_follower_pairs = [
    ('coach', 'player'),
    ('captain', 'crew'),
    ('boss', 'employee'),
    ('manager', 'staff'),
    ('director', 'actor'),
    ('supervisor', 'worker'),
    ('executive', 'employee'),
    ('foreperson', 'laborer'),
    ('chief', 'assistant'),
    ('general', 'soldier'),
    ('king', 'minister'),
    ('director', 'crew'),
    ('commander', 'soldier'),
    ('CEO', 'employee'),
    ('CFO', 'employee'),
    ('CTO', 'employee'),
    ('manager', 'operator')
]
analyze_word_pairs(get_clip_embedding, ("leader", "follower"), leader_follower_pairs, "clip")
# %% run: action/object pairs
action_object_pairs = [
    ('drive', 'car'),
    ('ride', 'bike'),
    ('read', 'book'),
    ('write', 'note'),
    ('draw', 'picture'),
    ('cook', 'meal'),
    ('eat', 'food'),
    ('drink', 'water'),
    ('play', 'game'),
    ('watch', 'movie'),
    ('open', 'door'),
    ('close', 'window'),
    ('kick', 'ball'),
    ('catch', 'ball'),
    ('wash', 'hands'),
    ('clean', 'room'),
    ('buy', 'ticket'),
    ('send', 'message'),
    ('charge', 'phone'),
    ('pack', 'bag')
]
analyze_word_pairs(get_clip_embedding, ("action", "object"), action_object_pairs, "clip")
# %% run: leader/follower pairs clustering
cluster_and_visualize_texts(leader_follower_pairs, get_clip_embedding, "clip")

# %% run: capital-country pairs
country_capital_pairs = [
    ('Afghanistan', 'Kabul'),
    ('Albania', 'Tirana'),
    ('Algeria', 'Algiers'),
    ('Andorra', 'Andorra la Vella'),
    ('Angola', 'Luanda'),
    ('Antigua and Barbuda', 'St. John’s'),
    ('Argentina', 'Buenos Aires'),
    ('Armenia', 'Yerevan'),
    ('Australia', 'Canberra'),
    ('Austria', 'Vienna'),
    ('Azerbaijan', 'Baku'),

    ('Bahamas', 'Nassau'),
    ('Bahrain', 'Manama'),
    ('Bangladesh', 'Dhaka'),
    ('Barbados', 'Bridgetown'),
    ('Belarus', 'Minsk'),
    ('Belgium', 'Brussels'),
    ('Belize', 'Belmopan'),
    ('Benin', 'Porto-Novo'),
    ('Bhutan', 'Thimphu'),
    ('Bolivia', 'Sucre'),
    ('Bosnia and Herzegovina', 'Sarajevo'),
    ('Botswana', 'Gaborone'),
    ('Brazil', 'Brasília'),
    ('Brunei', 'Bandar Seri Begawan'),
    ('Bulgaria', 'Sofia'),
    ('Burkina Faso', 'Ouagadougou'),
    ('Burundi', 'Gitega'),

    ('Cabo Verde', 'Praia'),
    ('Cambodia', 'Phnom Penh'),
    ('Cameroon', 'Yaoundé'),
    ('Canada', 'Ottawa'),
    ('Central African Republic', 'Bangui'),
    ('Chad', 'N’Djamena'),
    ('Chile', 'Santiago'),
    ('China', 'Beijing'),
    ('Colombia', 'Bogotá'),
    ('Comoros', 'Moroni'),
    ('Congo (Republic of the Congo)', 'Brazzaville'),
    ('Congo (Democratic Republic of the Congo)', 'Kinshasa'),
    ('Costa Rica', 'San José'),
    ('Croatia', 'Zagreb'),
    ('Cuba', 'Havana'),
    ('Cyprus', 'Nicosia'),
    ('Czech Republic', 'Prague'),

    ('Denmark', 'Copenhagen'),
    ('Djibouti', 'Djibouti'),
    ('Dominica', 'Roseau'),
    ('Dominican Republic', 'Santo Domingo'),

    ('Ecuador', 'Quito'),
    ('Egypt', 'Cairo'),
    ('El Salvador', 'San Salvador'),
    ('Equatorial Guinea', 'Malabo'),
    ('Eritrea', 'Asmara'),
    ('Estonia', 'Tallinn'),
    ('Eswatini', 'Mbabane'),
    ('Ethiopia', 'Addis Ababa'),

    ('Fiji', 'Suva'),
    ('Finland', 'Helsinki'),
    ('France', 'Paris'),

    ('Gabon', 'Libreville'),
    ('Gambia', 'Banjul'),
    ('Georgia', 'Tbilisi'),
    ('Germany', 'Berlin'),
    ('Ghana', 'Accra'),
    ('Greece', 'Athens'),
    ('Grenada', 'St. George’s'),
    ('Guatemala', 'Guatemala City'),
    ('Guinea', 'Conakry'),
    ('Guinea-Bissau', 'Bissau'),
    ('Guyana', 'Georgetown'),

    ('Haiti', 'Port-au-Prince'),
    ('Honduras', 'Tegucigalpa'),
    ('Hungary', 'Budapest'),

    ('Iceland', 'Reykjavík'),
    ('India', 'New Delhi'),
    ('Indonesia', 'Jakarta'),
    ('Iran', 'Tehran'),
    ('Iraq', 'Baghdad'),
    ('Ireland', 'Dublin'),
    ('Israel', 'Jerusalem'),
    ('Italy', 'Rome'),

    ('Jamaica', 'Kingston'),
    ('Japan', 'Tokyo'),
    ('Jordan', 'Amman'),

    ('Kazakhstan', 'Astana'),
    ('Kenya', 'Nairobi'),
    ('Kiribati', 'South Tarawa'),
    ('Kuwait', 'Kuwait City'),
    ('Kyrgyzstan', 'Bishkek'),

    ('Laos', 'Vientiane'),
    ('Latvia', 'Riga'),
    ('Lebanon', 'Beirut'),
    ('Lesotho', 'Maseru'),
    ('Liberia', 'Monrovia'),
    ('Libya', 'Tripoli'),
    ('Liechtenstein', 'Vaduz'),
    ('Lithuania', 'Vilnius'),
    ('Luxembourg', 'Luxembourg'),

    ('Madagascar', 'Antananarivo'),
    ('Malawi', 'Lilongwe'),
    ('Malaysia', 'Kuala Lumpur'),
    ('Maldives', 'Malé'),
    ('Mali', 'Bamako'),
    ('Malta', 'Valletta'),
    ('Marshall Islands', 'Majuro'),
    ('Mauritania', 'Nouakchott'),
    ('Mauritius', 'Port Louis'),
    ('Mexico', 'Mexico City'),
    ('Micronesia', 'Palikir'),
    ('Moldova', 'Chișinău'),
    ('Monaco', 'Monaco'),
    ('Mongolia', 'Ulaanbaatar'),
    ('Montenegro', 'Podgorica'),
    ('Morocco', 'Rabat'),
    ('Mozambique', 'Maputo'),
    ('Myanmar', 'Naypyidaw'),

    ('Namibia', 'Windhoek'),
    ('Nauru', 'Yaren'),
    ('Nepal', 'Kathmandu'),
    ('Netherlands', 'Amsterdam'),
    ('New Zealand', 'Wellington'),
    ('Nicaragua', 'Managua'),
    ('Niger', 'Niamey'),
    ('Nigeria', 'Abuja'),
    ('North Korea', 'Pyongyang'),
    ('North Macedonia', 'Skopje'),
    ('Norway', 'Oslo'),

    ('Oman', 'Muscat'),

    ('Pakistan', 'Islamabad'),
    ('Palau', 'Ngerulmud'),
    ('Panama', 'Panama City'),
    ('Papua New Guinea', 'Port Moresby'),
    ('Paraguay', 'Asunción'),
    ('Peru', 'Lima'),
    ('Philippines', 'Manila'),
    ('Poland', 'Warsaw'),
    ('Portugal', 'Lisbon'),

    ('Qatar', 'Doha'),

    ('Romania', 'Bucharest'),
    ('Russia', 'Moscow'),
    ('Rwanda', 'Kigali'),

    ('Saint Kitts and Nevis', 'Basseterre'),
    ('Saint Lucia', 'Castries'),
    ('Saint Vincent and the Grenadines', 'Kingstown'),
    ('Samoa', 'Apia'),
    ('San Marino', 'San Marino'),
    ('Sao Tome and Principe', 'São Tomé'),
    ('Saudi Arabia', 'Riyadh'),
    ('Senegal', 'Dakar'),
    ('Serbia', 'Belgrade'),
    ('Seychelles', 'Victoria'),
    ('Sierra Leone', 'Freetown'),
    ('Singapore', 'Singapore'),
    ('Slovakia', 'Bratislava'),
    ('Slovenia', 'Ljubljana'),
    ('Solomon Islands', 'Honiara'),
    ('Somalia', 'Mogadishu'),
    ('South Africa', 'Pretoria'),
    ('South Korea', 'Seoul'),
    ('South Sudan', 'Juba'),
    ('Spain', 'Madrid'),
    ('Sri Lanka', 'Sri Jayawardenepura Kotte'),
    ('Sudan', 'Khartoum'),
    ('Suriname', 'Paramaribo'),
    ('Sweden', 'Stockholm'),
    ('Switzerland', 'Bern'),
    ('Syria', 'Damascus'),

    ('Tajikistan', 'Dushanbe'),
    ('Tanzania', 'Dodoma'),
    ('Thailand', 'Bangkok'),
    ('Timor-Leste', 'Dili'),
    ('Togo', 'Lomé'),
    ('Tonga', 'Nukuʻalofa'),
    ('Trinidad and Tobago', 'Port of Spain'),
    ('Tunisia', 'Tunis'),
    ('Turkey', 'Ankara'),
    ('Turkmenistan', 'Ashgabat'),
    ('Tuvalu', 'Funafuti'),

    ('Uganda', 'Kampala'),
    ('Ukraine', 'Kyiv'),
    ('United Arab Emirates', 'Abu Dhabi'),
    ('United Kingdom', 'London'),
    ('United States', 'Washington, D.C.'),
    ('Uruguay', 'Montevideo'),
    ('Uzbekistan', 'Tashkent'),

    ('Vanuatu', 'Port Vila'),
    ('Vatican City', 'Vatican City'),
    ('Venezuela', 'Caracas'),
    ('Vietnam', 'Hanoi'),

    ('Yemen', 'Sana’a'),

    ('Zambia', 'Lusaka'),
    ('Zimbabwe', 'Harare'),
]
analyze_word_pairs(get_clip_embedding, ("capital", "country"), country_capital_pairs, "clip")
# reverse testing
analyze_word_pairs(get_clip_embedding, ("country", "capital"), country_capital_pairs, "clip")
# %% run: element/symbol pairs
element_symbol_pairs = [
    ('Hydrogen', 'H'),
    ('Helium', 'He'),
    ('Lithium', 'Li'),
    ('Beryllium', 'Be'),
    ('Boron', 'B'),
    ('Carbon', 'C'),
    ('Nitrogen', 'N'),
    ('Oxygen', 'O'),
    ('Fluorine', 'F'),
    ('Neon', 'Ne'),

    ('Sodium', 'Na'),
    ('Magnesium', 'Mg'),
    ('Aluminum', 'Al'),
    ('Silicon', 'Si'),
    ('Phosphorus', 'P'),
    ('Sulfur', 'S'),
    ('Chlorine', 'Cl'),
    ('Argon', 'Ar'),

    ('Potassium', 'K'),
    ('Calcium', 'Ca'),
    ('Scandium', 'Sc'),
    ('Titanium', 'Ti'),
    ('Vanadium', 'V'),
    ('Chromium', 'Cr'),
    ('Manganese', 'Mn'),
    ('Iron', 'Fe'),
    ('Cobalt', 'Co'),
    ('Nickel', 'Ni'),
    ('Copper', 'Cu'),
    ('Zinc', 'Zn'),
    ('Gallium', 'Ga'),
    ('Germanium', 'Ge'),
    ('Arsenic', 'As'),
    ('Selenium', 'Se'),
    ('Bromine', 'Br'),
    ('Krypton', 'Kr'),

    ('Rubidium', 'Rb'),
    ('Strontium', 'Sr'),
    ('Yttrium', 'Y'),
    ('Zirconium', 'Zr'),
    ('Niobium', 'Nb'),
    ('Molybdenum', 'Mo'),
    ('Technetium', 'Tc'),
    ('Ruthenium', 'Ru'),
    ('Rhodium', 'Rh'),
    ('Palladium', 'Pd'),
    ('Silver', 'Ag'),
    ('Cadmium', 'Cd'),
    ('Indium', 'In'),
    ('Tin', 'Sn'),
    ('Antimony', 'Sb'),
    ('Tellurium', 'Te'),
    ('Iodine', 'I'),
    ('Xenon', 'Xe'),

    ('Cesium', 'Cs'),
    ('Barium', 'Ba'),
    ('Lanthanum', 'La'),
    ('Cerium', 'Ce'),
    ('Praseodymium', 'Pr'),
    ('Neodymium', 'Nd'),
    ('Promethium', 'Pm'),
    ('Samarium', 'Sm'),
    ('Europium', 'Eu'),
    ('Gadolinium', 'Gd'),
    ('Terbium', 'Tb'),
    ('Dysprosium', 'Dy'),
    ('Holmium', 'Ho'),
    ('Erbium', 'Er'),
    ('Thulium', 'Tm'),
    ('Ytterbium', 'Yb'),
    ('Lutetium', 'Lu'),

    ('Hafnium', 'Hf'),
    ('Tantalum', 'Ta'),
    ('Tungsten', 'W'),
    ('Rhenium', 'Re'),
    ('Osmium', 'Os'),
    ('Iridium', 'Ir'),
    ('Platinum', 'Pt'),
    ('Gold', 'Au'),
    ('Mercury', 'Hg'),
    ('Thallium', 'Tl'),
    ('Lead', 'Pb'),
    ('Bismuth', 'Bi'),
    ('Polonium', 'Po'),
    ('Astatine', 'At'),
    ('Radon', 'Rn'),

    ('Francium', 'Fr'),
    ('Radium', 'Ra'),
    ('Actinium', 'Ac'),
    ('Thorium', 'Th'),
    ('Protactinium', 'Pa'),
    ('Uranium', 'U'),
    ('Neptunium', 'Np'),
    ('Plutonium', 'Pu'),
    ('Americium', 'Am'),
    ('Curium', 'Cm'),
    ('Berkelium', 'Bk'),
    ('Californium', 'Cf'),
    ('Einsteinium', 'Es'),
    ('Fermium', 'Fm'),
    ('Mendelevium', 'Md'),
    ('Nobelium', 'No'),
    ('Lawrencium', 'Lr'),

    ('Rutherfordium', 'Rf'),
    ('Dubnium', 'Db'),
    ('Seaborgium', 'Sg'),
    ('Bohrium', 'Bh'),
    ('Hassium', 'Hs'),
    ('Meitnerium', 'Mt'),
    ('Darmstadtium', 'Ds'),
    ('Roentgenium', 'Rg'),
    ('Copernicium', 'Cn'),
    ('Nihonium', 'Nh'),
    ('Flerovium', 'Fl'),
    ('Moscovium', 'Mc'),
    ('Livermorium', 'Lv'),
    ('Tennessine', 'Ts'),
    ('Oganesson', 'Og')
]
analyze_word_pairs(get_clip_embedding, ("element", "symbol"), element_symbol_pairs, "clip")

# %% run: state-capital pairs
state_capital_pairs = [
    ('Alabama', 'Montgomery'),
    ('Alaska', 'Juneau'),
    ('Arizona', 'Phoenix'),
    ('Arkansas', 'Little Rock'),
    ('California', 'Sacramento'),
    ('Colorado', 'Denver'),
    ('Connecticut', 'Hartford'),
    ('Delaware', 'Dover'),
    ('Florida', 'Tallahassee'),
    ('Georgia', 'Atlanta'),

    ('Hawaii', 'Honolulu'),
    ('Idaho', 'Boise'),
    ('Illinois', 'Springfield'),
    ('Indiana', 'Indianapolis'),
    ('Iowa', 'Des Moines'),
    ('Kansas', 'Topeka'),
    ('Kentucky', 'Frankfort'),
    ('Louisiana', 'Baton Rouge'),
    ('Maine', 'Augusta'),
    ('Maryland', 'Annapolis'),

    ('Massachusetts', 'Boston'),
    ('Michigan', 'Lansing'),
    ('Minnesota', 'Saint Paul'),
    ('Mississippi', 'Jackson'),
    ('Missouri', 'Jefferson City'),
    ('Montana', 'Helena'),
    ('Nebraska', 'Lincoln'),
    ('Nevada', 'Carson City'),
    ('New Hampshire', 'Concord'),
    ('New Jersey', 'Trenton'),

    ('New Mexico', 'Santa Fe'),
    ('New York', 'Albany'),
    ('North Carolina', 'Raleigh'),
    ('North Dakota', 'Bismarck'),
    ('Ohio', 'Columbus'),
    ('Oklahoma', 'Oklahoma City'),
    ('Oregon', 'Salem'),
    ('Pennsylvania', 'Harrisburg'),
    ('Rhode Island', 'Providence'),
    ('South Carolina', 'Columbia'),

    ('South Dakota', 'Pierre'),
    ('Tennessee', 'Nashville'),
    ('Texas', 'Austin'),
    ('Utah', 'Salt Lake City'),
    ('Vermont', 'Montpelier'),
    ('Virginia', 'Richmond'),
    ('Washington', 'Olympia'),
    ('West Virginia', 'Charleston'),
    ('Wisconsin', 'Madison'),
    ('Wyoming', 'Cheyenne'),
]
analyze_word_pairs(get_clip_embedding, ("state", "capital"), state_capital_pairs, "clip")
# reverse testing
analyze_word_pairs(get_clip_embedding, ("capital", "state"), state_capital_pairs, "clip")
# %% run: current/past pairs
current_past_pairs = [
    ('go', 'went'),
    ('come', 'came'),
    ('see', 'saw'),
    ('eat', 'ate'),
    ('drink', 'drank'),
    ('take', 'took'),
    ('give', 'gave'),
    ('make', 'made'),
    ('find', 'found'),
    ('think', 'thought'),

    ('write', 'wrote'),
    ('read', 'read'),
    ('speak', 'spoke'),
    ('run', 'ran'),
    ('sit', 'sat'),
    ('stand', 'stood'),
    ('bring', 'brought'),
    ('buy', 'bought'),
    ('catch', 'caught'),
    ('feel', 'felt'),

    ('keep', 'kept'),
    ('sleep', 'slept'),
    ('build', 'built'),
    ('drive', 'drove'),
    ('break', 'broke'),
    ('choose', 'chose'),
    ('fall', 'fell'),
    ('fly', 'flew'),
    ('grow', 'grew'),
    ('know', 'knew'),
]
analyze_word_pairs(get_clip_embedding, ("current", "past"), current_past_pairs, "clip")

# %%  Permutation test: shuffle second words while keeping first words fixed
import random

def analyze_word_pairs_with_permutation(get_embedding_func, anchor_pair, word_pairs, model_name=""):
    """
    Run analyze_word_pairs twice: once with original pairs, once with shuffled second words.
    This tests whether the relationship is specific to the pairs or just a property of the words.
    """
    # First run with original pairs
    print("=" * 80)
    print("ORIGINAL PAIRS:")
    print("=" * 80)
    analyze_word_pairs(get_embedding_func, anchor_pair, word_pairs, model_name)
    
    # Create permuted pairs: keep first words fixed, shuffle second words
    first_words = [pair[0] for pair in word_pairs]
    second_words = [pair[1] for pair in word_pairs]
    shuffled_second_words = second_words.copy()
    random.shuffle(shuffled_second_words)
    
    permuted_pairs = list(zip(first_words, shuffled_second_words))
    
    # Second run with permuted pairs
    print("\n" + "=" * 80)
    print("PERMUTED PAIRS (second words shuffled):")
    print("=" * 80)
    analyze_word_pairs(get_embedding_func, anchor_pair, permuted_pairs, f"{model_name}_permuted")

# Run permutation test on current/past pairs
analyze_word_pairs_with_permutation(get_clip_embedding, ("country", "capital"), country_capital_pairs, "clip")

# %% diff norm test
def analyze_word_pairs_diff_norm(get_embedding_func, anchor_pair, word_pairs, model_name=""):
    """
    Analyze word pairs by plotting custom norm vs norm of difference between pairs.

    For each pair (word1, word2): x-axis is the norm of the average of (word2-anchor2) and (word1-anchor1),
    y-axis is the norm of (word2 - word1).
    
    Args:
        get_embedding_func: Function to get embeddings for words
        anchor_pair: Tuple of (anchor1, anchor2) - the reference pair (e.g., ("man", "woman"))
        word_pairs: List of (word1, word2) tuples to analyze
        model_name: Name of the model for labeling
    """
    custom_norms = []
    diff_norms = []
    pair_labels = []

    # Get embeddings for anchor words
    anchor1, anchor2 = anchor_pair
    try:
        anchor1_emb = get_embedding_func(anchor1)
        anchor2_emb = get_embedding_func(anchor2)
    except KeyError:
        print(f"Warning: {anchor1} or {anchor2} not in vocabulary. Aborting analysis.")
        return

    for word1, word2 in word_pairs:
        try:
            w1_emb = get_embedding_func(word1)
            w2_emb = get_embedding_func(word2)
            
            # Extract context from phrases like "woman at office" -> "office"
            if ' at ' in word1:
                context_text = word1.split(' at ')[1]

                context_emb = get_embedding_func(f'person at {context_text}')
                custom_norm = np.linalg.norm(context_emb)
                x_title = 'Norm of context'
            else:
                # norm of average of (word2-anchor2) and (word1-anchor1)
                w1_minus_anchor = w1_emb - anchor1_emb
                w2_minus_anchor = w2_emb - anchor2_emb
                avg_vec = (w1_minus_anchor + w2_minus_anchor) / 2
                custom_norm = np.linalg.norm(avg_vec)
                x_title = f'Norm of average: (word2-{anchor2} + word1-{anchor1})/2'

            # Norm of difference between word2 and word1
            diff_norm = np.linalg.norm(w2_emb - w1_emb)

            custom_norms.append(custom_norm)
            diff_norms.append(diff_norm)
            pair_labels.append(f"{word1}-{word2}")
        except KeyError:
            print(f"Warning: Skipping pair ({word1}, {word2}) - word not found in vocabulary")
            continue

    if len(custom_norms) == 0:
        print("No valid word pairs found!")
        return

    # Correlation coefficient
    if len(custom_norms) > 1:
        correlation = np.corrcoef(custom_norms, diff_norms)[0,1]
    else:
        correlation = float('nan')

    # Plot: Custom norm vs difference norm
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, diff_norms, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], diff_norms[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel(x_title)
    plt.ylabel(f'Norm of difference (word2 - word1)')

    title_suffix = f" ({model_name})" if model_name else ""
    plt.title(f'Word Pairs (anchor: {anchor1}-{anchor2}): Custom Norm vs Difference Norm{title_suffix}\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename_suffix = f"_{model_name}" if model_name else ""
    plt.savefig(f'word_pairs_customnorm_vs_diffnorm{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nWord Pair Analysis (Diff Norm){title_suffix} (anchor: {anchor1}-{anchor2}):")
    print("-" * 70)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Custom norm: {custom_norms[i]:.4f} | Diff norm: {diff_norms[i]:.4f}")
    print(f"\nCorrelation (custom norm vs diff norm): {correlation:.4f}")

analyze_word_pairs_diff_norm(get_clip_embedding, ("country", "capital"), country_capital_pairs, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("man", "woman"), gender_context_pairs_combinational, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("state", "capital"), state_capital_pairs, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("adult", "young"), adult_young_pairs, "clip")
# %% run: place-person pairs
place_person_pairs = [
    ('teacher', 'school'),
    ('student', 'school'),
    ('professor', 'university'),
    ('librarian', 'library'),
    ('curator', 'museum'),
    ('banker', 'bank'),
    ('cashier', 'supermarket'),
    ('bartender', 'bar'),
    ('waiter', 'restaurant'),
    ('chef', 'restaurant'),
    ('baker', 'bakery'),
    ('barista', 'cafe'),
    ('doctor', 'hospital'),
    ('nurse', 'hospital'),
    ('dentist', 'clinic'),
    ('pharmacist', 'pharmacy'),
    ('pastor', 'church'),
    ('monk', 'monastery'),
    ('judge', 'court'),
    ('lawyer', 'court'),
    ('soldier', 'barracks'),
    ('guard', 'prison'),
    ('warden', 'prison'),
    ('farmer', 'farm'),
    ('rancher', 'ranch'),
    ('sailor', 'ship'),
    ('mechanic', 'garage'),
    ('pilot', 'airport'),
    ('concierge', 'hotel'),
    ('maid', 'hotel'),
    ('janitor', 'school'),
    ('coach', 'stadium'),
    ('athlete', 'stadium'),
    ('referee', 'stadium'),
    ('vendor', 'market'),
    ('clerk', 'office'),
    ('manager', 'office'),
    ('engineer', 'factory'),
    ('worker', 'factory'),
    ('actor', 'theater'),
    ('reporter', 'newsroom'),
    ('editor', 'newsroom'),
    ('senator', 'senate'),
    ('prisoner', 'prison'),
    ('inmate', 'prison'),
]
analyze_word_pairs(get_clip_embedding, ("place", "person"), place_person_pairs, "clip")
# %% run: place-person pairs clustering
cluster_and_visualize_texts(place_person_pairs, get_clip_embedding, "clip")
# %% run: red-green pairs
red_green_pairs = [
    ('strawberry', 'kiwi'),
    ('cherry', 'lime'),
    ('pomegranate', 'avocado'),
    ('raspberry', 'grape'),
    ('watermelon', 'honeydew'),
    ('plum', 'pear'),

    ('tomato', 'cucumber'),
    ('beet', 'broccoli'),
    ('radish', 'spinach'),
    ('chili', 'jalapeno'),
    ('cabbage', 'lettuce'),

    ('poppy', 'fern'),
    ('ladybug', 'grasshopper'),
    ('coral', 'seaweed'),
    ('lava', 'moss')
]
analyze_word_pairs(get_clip_embedding, ("red", "green"), red_green_pairs, "clip")
# %% run: soft-hard pairs
soft_hard_pairs = [
    ('cotton', 'stone'),
    ('sponge', 'brick'),
    ('pillow', 'metal'),
    ('fur', 'shell'),
    ('foam', 'rock'),
    ('rubber', 'glass'),
    ('cloth', 'concrete'),
    ('feather', 'armor'),
    ('moss', 'granite'),
    ('gel', 'steel'),
    ('mud', 'marble'),
    ('dough', 'iron'),
    ('bread', 'tile'),
    ('clay', 'pebble'),
    ('wax', 'ceramic'),
    ('velvet', 'plastic'),
    ('snow', 'ice'),      # snow softer than ice
    ('leaf', 'wood'),
    ('yarn', 'nail'),
    ('fur', 'bone')
]
analyze_word_pairs(get_clip_embedding, ("soft", "hard"), soft_hard_pairs, "clip")
# %% run: agent-tool pairs | not so ideal
agent_tool_pairs = [
    ('painter', 'brush'),
    ('writer', 'pen'),
    ('farmer', 'plow'),
    ('carpenter', 'hammer'),
    ('chef', 'knife'),
    ('baker', 'oven'),
    ('gardener', 'shovel'),
    ('fisher', 'net'),
    ('driver', 'wheel'),
    ('sailor', 'rope'),
    ('tailor', 'needle'),
    ('hunter', 'bow'),
    ('miner', 'pick'),
    ('barber', 'scissors'),
    ('photographer', 'camera'),
    ('mechanic', 'wrench'),
    ('shoemaker', 'awl'),
    ('potter', 'wheel'),
    ('weaver', 'loom'),
    ('mason', 'trowel')
]
analyze_word_pairs(get_clip_embedding, ("agent", "tool"), agent_tool_pairs, "clip")
# %% run solid-liquid pairs
solid_liquid_pairs = [
    ('ice', 'water'),
    ('salt', 'brine'),
    ('sand', 'slurry'),
    ('wax', 'oil'),
    ('butter', 'cream'),
    ('cheese', 'milk'),
    ('honeycomb', 'honey'),
    ('snow', 'rain'),
    ('soil', 'mud'),
    ('gel', 'solution'),
    ('crystal', 'fluid'),
    ('clay', 'slime'),
    ('metal', 'melt'),
    ('ash', 'broth'),
    ('powder', 'syrup'),
    ('pebble', 'stream'),
    ('resin', 'sap'),
    ('grain', 'soup'),
    ('granule', 'liquid'),
    ('rock', 'lava')
]
analyze_word_pairs(get_clip_embedding, ("solid", "liquid"), solid_liquid_pairs, "clip")
# %% run: fast-slow pairs
fast_slow_pairs = [
    ('cheetah', 'turtle'),
    ('falcon', 'snail'),
    ('rabbit', 'sloth'),
    ('shark', 'jellyfish'),
    ('horse', 'cow'),
    ('hawk', 'duck'),
    ('dolphin', 'manatee'),
    ('greyhound', 'pig'),
    ('antelope', 'sheep'),
    ('wasp', 'beetle'),
    ('squirrel', 'hedgehog'),
    ('dragonfly', 'moth')
]
analyze_word_pairs(get_clip_embedding, ("fast", "slow"), fast_slow_pairs, "clip")
# %% run: light-heavy pairs
light_heavy_pairs = [
    ('feather', 'stone'),
    ('leaf', 'brick'),
    ('paper', 'iron'),
    ('foam', 'concrete'),
    ('cotton', 'metal'),
    ('yarn', 'marble'),
    ('twig', 'log'),
    ('petal', 'rock'),
    ('balloon', 'anvil'),
    ('straw', 'steel'),
    ('cloth', 'granite'),
    ('seed', 'boulder')
]
analyze_word_pairs(get_clip_embedding, ("light", "heavy"), light_heavy_pairs, "clip")
# %% run bright-dark pairs
bright_dark_pairs = [
    ('sunflower', 'bat'),
    ('flame', 'shadow'),
    ('snow', 'coal'),
    ('pearl', 'obsidian'),
    ('mirror', 'soot'),
    ('ice', 'tar'),
    ('lightning', 'cave'),
    ('moon', 'fog'),
    ('chalk', 'charcoal'),
    ('cloud', 'smoke'),
    ('salt', 'mud'),
    ('paper', 'ink')
]
analyze_word_pairs(get_clip_embedding, ("bright", "dark"), bright_dark_pairs, "clip")

# %% run: hot-cold pairs
hot_cold_pairs = [
    ('lava', 'ice'),
    ('steam', 'frost'),
    ('ember', 'snow'),
    ('fire', 'glacier'),
    ('desert', 'tundra'),
    ('sun', 'hail'),
    ('pepper', 'mint'),
    ('soup', 'slush'),
    ('tea', 'water'),
    ('oven', 'freezer'),
    ('curry', 'cucumber'),
    ('coal', 'iceberg')
]
analyze_word_pairs(get_clip_embedding, ("hot", "cold"), hot_cold_pairs, "clip")
# %% run: good-bad pairs | not so ideal
good_evil_pairs = [
    ('angel', 'demon'),
    ('hero', 'villain'),
    ('saint', 'devil'),
    ('guardian', 'monster'),
    ('spirit', 'ghost'),
    ('phoenix', 'dragon'),
    ('light', 'shadow'),
    ('sun', 'storm'),
    ('dove', 'vulture'),
    ('unicorn', 'serpent'),
    ('guardian', 'fiend'),
    ('knight', 'bandit'),
    ('healer', 'poison'),
    ('candle', 'smoke'),
    ('harbor', 'reef'),
    ('pearl', 'fang'),
    ('lotus', 'thorn'),
    ('river', 'swamp'),
    ('spring', 'plague'),
    ('eden', 'abyss')
]
analyze_word_pairs(get_clip_embedding, ("good", "bad"), good_evil_pairs, "clip")
cluster_and_visualize_texts(good_evil_pairs, get_clip_embedding, "clip")

# %% run: happy-sad pairs | not so ideal
happy_sad_pairs = [
    ('sun', 'rain'),
    ('smile', 'frown'),
    ('laugh', 'cry'),
    ('joy', 'sorrow'),
    ('love', 'hate'),
    ('hope', 'despair'),
    ('peace', 'war'),
    ('pleasure', 'pain'),
    ('happiness', 'sadness')
]
analyze_word_pairs(get_clip_embedding, ("happy", "sad"), happy_sad_pairs, "clip")

# %% run: good-bad pairs | better than before
good_evil_pairs = [

    # ------------------------------
    # Mythic / Heroic beings (20 pairs)
    # ------------------------------
    ('angel', 'demon'),
    ('saint', 'devil'),
    ('hero', 'villain'),
    ('guardian', 'fiend'),
    ('knight', 'bandit'),
    ('paladin', 'raider'),
    ('protector', 'invader'),
    ('sage', 'trickster'),
    ('spirit', 'ghost'),
    ('oracle', 'warlock'),
    ('cherub', 'imp'),
    ('seraph', 'wraith'),
    ('monk', 'ogre'),
    ('seer', 'phantom'),
    ('champion', 'outlaw'),
    ('healer', 'poisoner'),
    ('wisp', 'ghoul'),
    ('phoenix', 'dragon'),
    ('unicorn', 'serpent'),
    ('griffin', 'harpy'),

    # ------------------------------
    # Animals as symbolic valence (15 pairs)
    # ------------------------------
    ('dove', 'vulture'),
    ('lamb', 'wolf'),
    ('robin', 'crow'),
    ('sparrow', 'raven'),
    ('butterfly', 'hornet'),
    ('swan', 'hyena'),
    ('deer', 'jackal'),
    ('pony', 'boar'),
    ('antelope', 'leopard'),
    ('gazelle', 'tiger'),
    ('finch', 'hawk'),
    ('heron', 'python'),
    ('otter', 'rat'),
    ('beetle', 'wasp'),
    ('seal', 'shark'),

    # ------------------------------
    # Plants / natural forms symbolic (12 pairs)
    # ------------------------------
    ('lotus', 'thorn'),
    ('rose', 'thistle'),
    ('lavender', 'nettles'),
    ('willow', 'bramble'),
    ('fern', 'cactus'),
    ('ivy', 'weed'),
    ('mint', 'fungus'),
    ('moss', 'mold'),
    ('bamboo', 'brier'),
    ('orchid', 'burdock'),
    ('peony', 'ragweed'),
    ('cypress', 'poison'),

    # ------------------------------
    # Natural phenomena (12 pairs)
    # ------------------------------
    ('sun', 'storm'),
    ('breeze', 'blizzard'),
    ('rainbow', 'thunder'),
    ('spring', 'plague'),
    ('dawn', 'midnight'),
    ('harbor', 'reef'),
    ('river', 'swamp'),
    ('oasis', 'desert'),
    ('meadow', 'wasteland'),
    ('stream', 'quicksand'),
    ('spark', 'smoke'),
    ('flame', 'ash'),

    # ------------------------------
    # Objects with valence symbolism (12 pairs)
    # ------------------------------
    ('candle', 'shadow'),
    ('pearl', 'fang'),
    ('jewel', 'rust'),
    ('harp', 'dagger'),
    ('mirror', 'shard'),
    ('lantern', 'chain'),
    ('banner', 'trap'),
    ('cloak', 'snare'),
    ('bell', 'blade'),
    ('cup', 'spike'),
    ('scroll', 'curse'),
    ('key', 'lock'),

    # ------------------------------
    # Abstract symbolic entities (13 pairs)
    # ------------------------------
    ('hope', 'fear'),
    ('mercy', 'wrath'),
    ('grace', 'vice'),
    ('truth', 'lie'),
    ('faith', 'doubt'),
    ('order', 'chaos'),
    ('honor', 'shame'),
    ('valor', 'terror'),
    ('virtue', 'sin'),
    ('charity', 'greed'),
    ('joy', 'grief'),
    ('purity', 'corruption'),
    ('blessing', 'curse'),

    # ------------------------------
    # Place / realm symbolic (16 pairs)
    # ------------------------------
    ('eden', 'abyss'),
    ('haven', 'pit'),
    ('sanctum', 'crypt'),
    ('temple', 'ruin'),
    ('garden', 'grave'),
    ('village', 'dungeon'),
    ('harbor', 'reef'),
    ('meadow', 'bog'),
    ('island', 'labyrinth'),
    ('citadel', 'fortress'),   # fortress negative meaning here is "imposing/dangerous"
    ('grove', 'cavern'),
    ('porch', 'dungeon'),
    ('court', 'gallows'),
    ('chapel', 'catacomb'),
    ('bridge', 'chasm'),
    ('tower', 'prison')
]
analyze_word_pairs(get_clip_embedding, ("good", "bad"), good_evil_pairs, "clip")
cluster_and_visualize_texts(good_evil_pairs, get_clip_embedding, "clip")

# %% USstate-abbreviation pairs | works well
us_state_abbrev = [
('Alabama', 'AL'),
('Alaska', 'AK'),
('Arizona', 'AZ'),
('Arkansas', 'AR'),
('California', 'CA'),
('Colorado', 'CO'),
('Connecticut', 'CT'),
('Delaware', 'DE'),
('Florida', 'FL'),
('Georgia', 'GA'),
('Hawaii', 'HI'),
('Idaho', 'ID'),
('Illinois', 'IL'),
('Indiana', 'IN'),
('Iowa', 'IA'),
('Kansas', 'KS'),
('Kentucky', 'KY'),
('Louisiana', 'LA'),
('Maine', 'ME'),
('Maryland', 'MD'),
('Massachusetts', 'MA'),
('Michigan', 'MI'),
('Minnesota', 'MN'),
('Mississippi', 'MS'),
('Missouri', 'MO'),
('Montana', 'MT'),
('Nebraska', 'NE'),
('Nevada', 'NV'),
('New_Hampshire', 'NH'),
('New_Jersey', 'NJ'),
('New_Mexico', 'NM'),
('New_York', 'NY'),
('North_Carolina', 'NC'),
('North_Dakota', 'ND'),
('Ohio', 'OH'),
('Oklahoma', 'OK'),
('Oregon', 'OR'),
('Pennsylvania', 'PA'),
('Rhode_Island', 'RI'),
('South_Carolina', 'SC'),
('South_Dakota', 'SD'),
('Tennessee', 'TN'),
('Texas', 'TX'),
('Utah', 'UT'),
('Vermont', 'VT'),
('Virginia', 'VA'),
('Washington', 'WA'),
('West_Virginia', 'WV'),
('Wisconsin', 'WI'),
('Wyoming', 'WY')
]
analyze_word_pairs(get_clip_embedding, ("state", "abbreviation"), us_state_abbrev, "clip")
# %% country-currency pairs | this works, unlike the airport-code pairs, probably because currency is common
# seems to show the pattern of frequency
country_currency = [
    ('United_States', 'USD'),
    ('Canada', 'CAD'),
    ('Mexico', 'MXN'),
    ('Brazil', 'BRL'),
    ('United_Kingdom', 'GBP'),
    ('Eurozone', 'EUR'),
    ('Switzerland', 'CHF'),
    ('Russia', 'RUB'),
    ('Turkey', 'TRY'),
    ('Saudi_Arabia', 'SAR'),
    ('United_Arab_Emirates', 'AED'),
    ('India', 'INR'),
    ('China', 'CNY'),
    ('Japan', 'JPY'),
    ('South_Korea', 'KRW'),
    ('Australia', 'AUD'),
    ('New_Zealand', 'NZD'),
    ('South_Africa', 'ZAR'),
    ('Nigeria', 'NGN'),
    ('Egypt', 'EGP')
]
analyze_word_pairs(get_clip_embedding, ("country", "currency"), country_currency, "clip")
# %% run: capital-province pairs
china_province_capital = [
    ('Shijiazhuang', 'Hebei'),
    ('Taiyuan', 'Shanxi'),
    ('Hohhot', 'Inner_Mongolia'),
    ('Shenyang', 'Liaoning'),
    ('Changchun', 'Jilin'),
    ('Harbin', 'Heilongjiang'),
    ('Nanjing', 'Jiangsu'),
    ('Hangzhou', 'Zhejiang'),
    ('Hefei', 'Anhui'),
    ('Fuzhou', 'Fujian'),
    ('Nanchang', 'Jiangxi'),
    ('Jinan', 'Shandong'),
    ('Zhengzhou', 'Henan'),
    ('Wuhan', 'Hubei'),
    ('Changsha', 'Hunan'),
    ('Guangzhou', 'Guangdong'),
    ('Nanning', 'Guangxi'),
    ('Haikou', 'Hainan'),
    ('Chengdu', 'Sichuan'),
    ('Guiyang', 'Guizhou'),
    ('Kunming', 'Yunnan'),
    ('Xi’an', 'Shaanxi'),
    ('Lanzhou', 'Gansu'),
    ('Xining', 'Qinghai'),
    ('Yinchuan', 'Ningxia')
]
analyze_word_pairs(get_clip_embedding, ("capital", "province"), china_province_capital, "clip")
# %% run: airport-code pairs | not working at all, probably because of the special characters
us_airport_code_airportname_pairs = [
    ('LAX', 'Los Angeles International Airport'),
    ('JFK', 'John F. Kennedy International Airport'),
    ('SFO', 'San Francisco International Airport'),
    ('ORD', 'O’Hare International Airport'),
    ('ATL', 'Hartsfield–Jackson Atlanta International Airport'),
    ('SEA', 'Seattle–Tacoma International Airport'),
    ('MIA', 'Miami International Airport'),
    ('DFW', 'Dallas/Fort Worth International Airport'),
    ('DEN', 'Denver International Airport'),
    ('BOS', 'Logan International Airport'),
    ('IAD', 'Washington Dulles International Airport'),
    ('DCA', 'Ronald Reagan Washington National Airport'),
    ('PHX', 'Phoenix Sky Harbor International Airport'),
    ('LAS', 'Harry Reid International Airport'),
    ('CLT', 'Charlotte Douglas International Airport'),
    ('IAH', 'George Bush Intercontinental Airport'),
    ('HNL', 'Daniel K. Inouye International Airport'),
    ('DTW', 'Detroit Metropolitan Airport'),
    ('MSP', 'Minneapolis–Saint Paul International Airport'),
    ('PHL', 'Philadelphia International Airport')
]
analyze_word_pairs(get_clip_embedding, ("code", "airport"), us_airport_code_airportname_pairs, "clip")
# %% run: animal-voice pairs
animal_voice_pairs = [
    ('dog', 'bark'),
    ('cat', 'meow'),
    ('cow', 'moo'),
    ('sheep', 'bleat'),
    ('goat', 'bleat'),
    ('horse', 'neigh'),
    ('pig', 'oink'),
    ('duck', 'quack'),
    ('chicken', 'cluck'),
    ('rooster', 'crow'),
    ('turkey', 'gobble'),
    ('goose', 'honk'),
    ('owl', 'hoot'),
    ('dove', 'coo'),
    ('pigeon', 'coo'),
    ('frog', 'croak'),
    ('toad', 'croak'),
    ('cricket', 'chirp'),
    ('grasshopper', 'chirp'),
    ('sparrow', 'chirp'),
    ('crow', 'caw'),
    ('raven', 'caw'),
    ('lion', 'roar'),
    ('tiger', 'roar'),
    ('bear', 'growl'),
    ('wolf', 'howl'),
    ('coyote', 'howl'),
    ('elephant', 'trumpet'),
    ('seal', 'bark'),
    ('dolphin', 'click'),
    ('whale', 'song'),
    ('bee', 'buzz'),
    ('fly', 'buzz'),
    ('mosquito', 'buzz'),
    ('snake', 'hiss'),
    ('swan', 'hiss'),
    ('penguin', 'honk'),
    ('parrot', 'squawk'),
    ('eagle', 'screech'),
    ('hawk', 'screech')
]
analyze_word_pairs(get_clip_embedding, ("animal", "voice"), animal_voice_pairs, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("animal", "voice"), animal_voice_pairs, "clip")
# %% important | function: base-vector test
# Create modified gender pairs with all female roles replaced by "female"
gender_context_pairs_semantic_modified = [
    (pair[0], 'female') for pair in gender_context_pairs_semantic
]
analyze_word_pairs(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic_modified, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic_modified, "clip")


# %% run: base-vector test
analyze_word_pairs(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic, "clip")
analyze_word_pairs_diff_norm(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic, "clip")


# %% function: unified function to test norm-shrink hypothesis
def analyze_word_pairs_with_normalization(get_embedding_func, anchor_pair, word_pairs, model_name="", normalize=False):
    """
    Analyze word pairs by plotting custom norm vs both difference norm and cosine similarity.
    
    Creates two plots:
    1. Custom norm vs difference norm (word2 - word1)
    2. Custom norm vs cosine similarity between word1 and word2
    
    Both plots include a horizontal dashed line indicating the anchor pair's metric value.
    
    Args:
        get_embedding_func: Function to get embeddings for words
        anchor_pair: Tuple of (anchor1, anchor2) - the reference pair (e.g., ("man", "woman"))
        word_pairs: List of (word1, word2) tuples to analyze
        model_name: Name of the model for labeling
        normalize: If True, normalize embeddings before computing Y-axis metrics (diff norm and cosine)
    """
    custom_norms = []
    diff_norms = []
    pair_cosine_sims = []
    pair_labels = []

    # Get embeddings for anchor words
    anchor1, anchor2 = anchor_pair
    try:
        anchor1_emb = get_embedding_func(anchor1)
        anchor2_emb = get_embedding_func(anchor2)
    except KeyError:
        print(f"Warning: {anchor1} or {anchor2} not in vocabulary. Aborting analysis.")
        return

    # Compute anchor metrics
    if normalize:
        anchor1_emb_norm = anchor1_emb / np.linalg.norm(anchor1_emb)
        anchor2_emb_norm = anchor2_emb / np.linalg.norm(anchor2_emb)
        anchor_diff_norm = np.linalg.norm(anchor2_emb_norm - anchor1_emb_norm)
        anchor_cosine_sim = np.dot(anchor1_emb_norm, anchor2_emb_norm)
    else:
        anchor_diff_norm = np.linalg.norm(anchor2_emb - anchor1_emb)
        anchor_cosine_sim = np.dot(anchor1_emb, anchor2_emb) / (np.linalg.norm(anchor1_emb) * np.linalg.norm(anchor2_emb))

    for word1, word2 in word_pairs:
        try:
            w1_emb = get_embedding_func(word1)
            w2_emb = get_embedding_func(word2)
            
            # X-axis: custom norm using original embeddings
            w1_minus_anchor = w1_emb - anchor1_emb
            w2_minus_anchor = w2_emb - anchor2_emb
            avg_vec = (w1_minus_anchor + w2_minus_anchor) / 2
            custom_norm = np.linalg.norm(avg_vec)
            x_title = f'Norm of average: (word2-{anchor2} + word1-{anchor1})/2'

            # Y-axis metrics: use normalized embeddings if requested
            if normalize:
                w1_emb_norm = w1_emb / np.linalg.norm(w1_emb)
                w2_emb_norm = w2_emb / np.linalg.norm(w2_emb)
                diff_norm = np.linalg.norm(w2_emb_norm - w1_emb_norm)
                pair_cosine_sim = np.dot(w1_emb_norm, w2_emb_norm)
            else:
                diff_norm = np.linalg.norm(w2_emb - w1_emb)
                pair_cosine_sim = np.dot(w1_emb, w2_emb) / (np.linalg.norm(w1_emb) * np.linalg.norm(w2_emb))

            custom_norms.append(custom_norm)
            diff_norms.append(diff_norm)
            pair_cosine_sims.append(pair_cosine_sim)
            pair_labels.append(f"{word1}-{word2}")
        except KeyError:
            print(f"Warning: Skipping pair ({word1}, {word2}) - word not found in vocabulary")
            continue

    if len(custom_norms) == 0:
        print("No valid word pairs found!")
        return

    # Compute correlations
    if len(custom_norms) > 1:
        correlation_diff = np.corrcoef(custom_norms, diff_norms)[0, 1]
        correlation_cosine = np.corrcoef(custom_norms, pair_cosine_sims)[0, 1]
    else:
        correlation_diff = float('nan')
        correlation_cosine = float('nan')

    title_suffix = f" ({model_name})" if model_name else ""
    filename_suffix = f"_{model_name}" if model_name else ""
    norm_suffix = "_normalized" if normalize else ""

    # Plot 1: Custom norm vs difference norm
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, diff_norms, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], diff_norms[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Add horizontal line for anchor difference norm
    plt.axhline(y=anchor_diff_norm, color='red', linestyle='--', linewidth=2, 
                label=f'Anchor ({anchor1}-{anchor2}) diff norm: {anchor_diff_norm:.4f}')

    plt.xlabel(x_title)
    plt.ylabel(f'Norm of difference (word2 - word1){"" if not normalize else " [normalized]"}')
    plt.title(f'Word Pairs (anchor: {anchor1}-{anchor2}): Custom Norm vs Difference Norm{title_suffix}\nCorrelation: {correlation_diff:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'word_pairs_customnorm_vs_diffnorm{filename_suffix}{norm_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Custom norm vs cosine similarity
    plt.figure(figsize=(12, 8))
    plt.scatter(custom_norms, pair_cosine_sims, s=100, alpha=0.6)

    # Add labels for each point
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (custom_norms[i], pair_cosine_sims[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Add horizontal line for anchor cosine similarity
    plt.axhline(y=anchor_cosine_sim, color='red', linestyle='--', linewidth=2,
                label=f'Anchor ({anchor1}-{anchor2}) cosine sim: {anchor_cosine_sim:.4f}')

    plt.xlabel(x_title)
    plt.ylabel(f'Cosine Similarity between word pairs{"" if not normalize else " [normalized]"}')
    plt.title(f'Word Pairs (anchor: {anchor1}-{anchor2}): Custom Norm vs Cosine Similarity{title_suffix}\nCorrelation: {correlation_cosine:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'word_pairs_customnorm_vs_cosine{filename_suffix}{norm_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nWord Pair Analysis{title_suffix} (anchor: {anchor1}-{anchor2}){'[normalized]' if normalize else ''}:")
    print("-" * 80)
    print(f"Anchor pair: {anchor1}-{anchor2}")
    print(f"  Anchor diff norm: {anchor_diff_norm:.4f}")
    print(f"  Anchor cosine sim: {anchor_cosine_sim:.4f}")
    print("-" * 80)
    for i, label in enumerate(pair_labels):
        print(f"{label:25s} | Custom norm: {custom_norms[i]:.4f} | Diff norm: {diff_norms[i]:.4f} | Cosine sim: {pair_cosine_sims[i]:.4f}")
    print(f"\nCorrelation (custom norm vs diff norm): {correlation_diff:.4f}")
    print(f"Correlation (custom norm vs cosine sim): {correlation_cosine:.4f}")

# %% run: unified testing, gender pairs
analyze_word_pairs_with_normalization(get_clip_embedding, ("man", "woman"), gender_context_pairs_semantic, "clip", normalize=True)
analyze_word_pairs_with_normalization(get_clip_embedding, ("man", "woman"), gender_context_pairs_semantic, "clip", normalize=False)

# %% run: unified testing, country-capital pairs
analyze_word_pairs_with_normalization(get_clip_embedding, ("country", "capital"), country_capital_pairs, "clip", normalize=True)

# %% run: place-person pairs
analyze_word_pairs_with_normalization(get_clip_embedding, ("place", "person"), place_person_pairs, "clip", normalize=True)
analyze_word_pairs_with_normalization(get_clip_embedding, ("place", "person"), place_person_pairs, "clip", normalize=False)

# %% run: state-capital pairs
analyze_word_pairs_with_normalization(get_clip_embedding, ("state", "capital"), state_capital_pairs, "clip", normalize=True)
analyze_word_pairs_with_normalization(get_clip_embedding, ("state", "capital"), state_capital_pairs, "clip", normalize=False)

# %% run: adult-young pairs
analyze_word_pairs_with_normalization(get_clip_embedding, ("adult", "young"), adult_young_pairs, "clip", normalize=True)
analyze_word_pairs_with_normalization(get_clip_embedding, ("adult", "young"), adult_young_pairs, "clip", normalize=False)
# %%
