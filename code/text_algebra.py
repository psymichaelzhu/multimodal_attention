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


# %% word pairs
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
    pca_cluster = PCA(n_components=min(16, len(all_texts) - 1))
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
analyze_word_pairs(get_clip_embedding, ("male", "female"), gender_context_pairs_semantic, "clip")
# reverse testing
analyze_word_pairs(get_clip_embedding, ("female", "male"), gender_context_pairs_semantic, "clip")


# %% run: man/woman location pairs
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
    ('goat', 'kid'),
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
# %%
analyze_gender_pairs(get_clip_embedding, gender_context_pairs_semantic, "clip")