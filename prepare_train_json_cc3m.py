import os
import json
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

data_dir = '/home/ubuntu/CC3M/cc3m-images'
save_dir = 'cc3m_process_json_768'

sample_each_folder = 10000  # images in each folder, may be smaller than 10000, some images are failed
sample_each_process_file = 100000
num_folders_each_process_file = sample_each_process_file // sample_each_folder

folders = sorted(glob.glob(os.path.join(data_dir, '*')))
num_process_files = len(folders) // num_folders_each_process_file

print(f'num_process_files: {num_process_files}, num_folders_each_process_file: {num_folders_each_process_file}')

def process_folder(folder):
    res = []
    images = os.listdir(folder)
    images = [os.path.join(folder, image) for image in images if image.endswith('.jpg')]
    prompts = [image_path.replace('.jpg', '.txt') for image_path in images]
    for image_path, prompt_path in zip(images, prompts):
        try:
            with open(prompt_path, 'r') as f:
                prompt = f.read()
            res.append({'image': image_path, 'caption': prompt})
        except Exception as e:
            print(f"Error processing {prompt_path}: {e}")
    return res

def process_files(start_index, end_index, process_file_id):
    os.makedirs(save_dir, exist_ok=True)
    process_file = os.path.join(save_dir, f'process_{process_file_id}.json')
    if os.path.exists(process_file):
        return
    results = []
    folders_this_process_file = folders[start_index:end_index]
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on your system's capabilities
        future_to_folder = {executor.submit(process_folder, folder): folder for folder in folders_this_process_file}
        for future in tqdm(as_completed(future_to_folder), total=len(folders_this_process_file)):
            folder_results = future.result()
            results.extend(folder_results)
    with open(process_file, 'w') as f:
        json.dump(results, f, indent=4)

# Adjust the number of workers for processing files based on your system's capabilities
with ThreadPoolExecutor(max_workers=5) as executor:
    for i in range(num_process_files):
        start_index = i * num_folders_each_process_file
        end_index = (i + 1) * num_folders_each_process_file
        executor.submit(process_files, start_index, end_index, i)
