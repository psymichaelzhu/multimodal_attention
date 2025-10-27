# %% objective
#  use multi-modal model to describe videos in natural language

# %% preparation
import os
import cv2
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
from PIL import Image
import time
from tqdm import tqdm

change_dir = "/Users/rezek_zhu/clip_social_annotation/code/selective_attention"
os.chdir(change_dir)
print(os.getcwd())

# %% helper function
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
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

# %% extract first frame for all videos
output_dir = "../../data/video/first_frame"

video_dir = "../../data/video/original_clips"
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

if False:
    os.makedirs(output_dir, exist_ok=True)
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        frame = get_first_frame(video_path)
        
        output_filename = os.path.splitext(video_file)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, frame)
else:
    print("Output frames already exist")

file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

#%% use multi-modal model to describe videos 

n_simulation = 5
model_name = "gpt-4.1-mini"
prompt = "Describe this picture in concise language:"

client = OpenAI()

results = pd.DataFrame(columns=['video_name', 'simulation_id', 'content'])

results_file = "../../data/video/descriptions.csv"
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    processed_videos = set(existing_results['video_name'])
else:
    existing_results = pd.DataFrame(columns=['video_name', 'simulation_id', 'content'])
    processed_videos = set()

for file_path in tqdm(file_paths[0:3], desc="Processing videos"):
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
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "file_id": file_id,
                                },
                            ],
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
                'simulation_id': sim,
                'content': description
            }

        pd.concat([existing_results, results]).to_csv(results_file, index=False)

    img = Image.open(file_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Video: {video_name}")
    plt.show()
    
    print(f"\nDescriptions for {video_name}:")
    for sim_id, desc in enumerate(video_descriptions):
        print(f"\nSimulation {sim_id}:")
        print(desc)
    
# %%
