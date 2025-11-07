# %% objective
#  use GPT-4.1-mini to annotate video through free association
# allowing varying n_simulation, n_word, and frame_index

# %% preparation
import os
import cv2
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
from PIL import Image
import time
from tqdm import tqdm

change_dir = "/Users/rezek_zhu/multimodal_attention"
os.chdir(change_dir)
print(os.getcwd())

# %% helper function
def extract_one_frame(video_path,frame_index=1,to_rgb=True):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 250))
        return frame
    return None


def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

# %% main function
def extract_frames(frame_index, video_dir="data/video/original_clip", overwrite=False):
    """
    Extract frames from all videos in video_dir at specified frame_index.
    
    Args:
        frame_index (int): Frame index to extract from videos
        video_dir (str): Directory containing source videos
        overwrite (bool): Whether to overwrite existing frames
        
    Returns:
        list: Paths to extracted frame images
    """
    output_dir = f"data/video/frame/{frame_index}"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in tqdm(video_files, desc=f"Extracting frame {frame_index}"):
        output_filename = os.path.splitext(video_file)[0] + f"@{frame_index}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        if overwrite or not os.path.exists(output_path):
            video_path = os.path.join(video_dir, video_file)
            frame = extract_one_frame(video_path, frame_index=frame_index, to_rgb=False)
            cv2.imwrite(output_path, frame) #cv2.imwrite expects BGR format

    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))]

def generate_free_associations(file_paths, frame_index, n_simulation=1, n_word=7, 
                             model_name="gpt-4.1-mini", show_result=False):
    """
    Generate free associations for video frames using a multi-modal model.
    
    Args:
        file_paths (list): List of paths to video frame images
        frame_index (int): Frame index being processed
        n_simulation (int): Number of simulations per frame
        n_word (int): Number of words to generate per simulation
        model_name (str): Name of the model to use
        show_result (bool): Whether to display results
        
    Returns:
        pd.DataFrame: Results containing video names, simulation IDs and generated content
    """
    prompt = f"List {n_word} words that best describe this scene (in descending order of relevance):"

    results = pd.DataFrame(columns=['video_name', 'simulation_id', 'content'])
    
    results_file = f"data/video/free_association/frame{frame_index}_word{n_word}_sim{n_simulation}.csv"
    
    if not os.path.exists(f"data/video/free_association"):
        os.makedirs(f"data/video/free_association")

    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        processed_videos = set(existing_results['video_name'])
    else:
        existing_results = pd.DataFrame(columns=['video_name', 'simulation_id', 'content'])
        processed_videos = set()

    for file_path in tqdm(file_paths, desc=f"Processing frame {frame_index}"):
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        
        video_descriptions = []
        if video_name in processed_videos:
            video_results = existing_results[existing_results['video_name'] == video_name]
            video_descriptions = video_results['content'].tolist()
        else:
            file_id = create_file(file_path)

            for sim in range(n_simulation):
                max_retries = 10
                base_delay = 1 
                for retry in range(max_retries):
                    try:
                        response = client.responses.create(
                            model=model_name,
                            input=[{
                                "role": "system",
                                "content": "Output only words separated by commas. No explanations, no extra text. Hyphenated compounds are allowed when needed."
                            },
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {"type": "input_image", "file_id": file_id}
                                ]
                            }],
                        )
                        break  
                    except Exception as e:
                        if retry == max_retries - 1:  
                            raise  
                        delay = base_delay * (2 ** retry)  
                        time.sleep(delay)
                        continue

                description = response.output_text.strip()
                video_descriptions.append(description)
                
                results.loc[len(results)] = {
                    'video_name': video_name,
                    'simulation_id': sim+1,
                    'content': description
                }

            pd.concat([existing_results, results]).to_csv(results_file, index=False)

        if show_result:
            img = Image.open(file_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Video: {video_name}")
            plt.show()
            
            print(f"\nDescriptions for {video_name}:")
            for sim_id, desc in enumerate(video_descriptions):
                print(f"\nSimulation {sim_id+1}:")
                print(desc)
                
    return results
    
# %% main
if __name__ == "__main__":
    client = OpenAI()
    for frame_index in [40]:
        for n_simulation in [50,10]:
            for n_word in [10,7,5,3]:
                frame_paths = extract_frames(frame_index, overwrite=False)
                results = generate_free_associations(frame_paths, frame_index, n_simulation, n_word, show_result=False)
                print(f"Frame {frame_index}, n_simulation {n_simulation}, n_word {n_word}, results: {results}")
# %%
print("Done")

# %%
