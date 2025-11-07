# %% objective
# to map free-association words to dimensions (character, relationship, environment, activity, object, emotional state, etc.)

# %% preparation
import pandas as pd
import os

change_dir = "/Users/rezek_zhu/multimodal_attention"
os.chdir(change_dir)
print(os.getcwd())

frame_index = 40
n_word = 10
n_sim = 50

# load data
dimension_words_df = pd.read_csv('data/video/gpt_dimension/dimension_words.csv')
free_association_df = pd.read_csv(f'data/video/free_association/frame{frame_index}_word{n_word}_sim{n_sim}.csv')


# %% process free-association data
# Split the content column into individual word columns
def process_free_association():
    processed_df = free_association_df.copy()

    word_columns = []
    for i in range(n_word):  
        word_columns.append(f'word_{i+1}')
    
    for col in word_columns:
        processed_df[col] = ''
    
    for idx, row in processed_df.iterrows():
        if pd.notna(row['content']):
            words = row['content'].split(',')
            # Only take up to n_word words
            words = words[:10]
            for i, word in enumerate(words):
                processed_df.at[idx, f'word_{i+1}'] = word.strip()
    
    processed_df = processed_df.drop('content', axis=1)
    
    return processed_df

# Map words to categories based on dimension_words
def map_words_to_categories(processed_df):
    category_columns = []
    for i in range(n_word):
        category_columns.append(f'category_{i+1}')
        processed_df[f'category_{i+1}'] = ''
    
    # Group dimension_words by video to create video-specific mappings
    video_dimensions = {}
    for _, row in dimension_words_df.iterrows():
        video = row['video']
        dimension = row['dimension']
        word = row['word']
        
        if video not in video_dimensions:
            video_dimensions[video] = {}
        
        if word not in video_dimensions[video]:
            video_dimensions[video][word] = dimension
    
    # Map each word to its category based on the video-specific mapping
    for idx, row in processed_df.iterrows():
        video_name = row['video_name']
        
        if video_name not in video_dimensions:
            continue
        
        word_to_dimension = video_dimensions[video_name]
        
        for i in range(10):
            word_col = f'word_{i+1}'
            cat_col = f'category_{i+1}'
            
            if pd.notna(row[word_col]) and row[word_col] in word_to_dimension:
                processed_df.at[idx, cat_col] = word_to_dimension[row[word_col]]
            else:
                processed_df.at[idx, cat_col] = 'not_found'
    
    return processed_df

# Main processing function
def process_and_save():
    # Split the free association data content column into individual word columns
    processed_df = process_free_association()
    
    # Map words to categories based on dimension_words_df
    final_df = map_words_to_categories(processed_df)
    
    # Save the processed dataframe
    output_file = f'data/video/free_association/processed_frame{frame_index}_word{n_word}_sim{n_sim}.csv'
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    final_df.to_csv(output_file, index=False)
    print(f"Processing complete. Output saved to 'data/video/free_association/processed_frame{frame_index}_word{n_word}_sim{n_sim}.csv'")
    
    return final_df

# Execute the processing
if __name__ == "__main__":
    process_and_save()

# %%
