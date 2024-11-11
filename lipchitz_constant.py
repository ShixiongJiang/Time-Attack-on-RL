import zipfile
import os
from stable_baselines3 import PPO, A2C, SAC, TD3
import torch
import scipy.io as sio
import numpy as np
import safety_gymnasium

# Define the path to your zip file and extraction directory
task_list = ['Goal', 'Push', 'Button', 'Race']
alg_list = [PPO, A2C, SAC, TD3]
for task in task_list:
    for alg in alg_list:
        alg_name = alg.__name__
#         zip_file_path = f'./model/SafetyPoint{task}0-{alg_name}.zip'
# #
#         if not os.path.exists(zip_file_path):
#             print("File exists!")
#             continue
#
#
#         extract_to_dir = f'./model/weights/{task}{alg_name}/'
#
#         # Extract the zip file
#         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_to_dir)
#
#         # List the extracted files
#         extracted_files = os.listdir(extract_to_dir)
#         print("Extracted files:", extracted_files)

        #
        # extracted_weights_dir = f'./model/weights/{task}{alg_name}/'
        # if not os.path.exists(extracted_weights_dir):
        #     continue
        # output_mat_dir = f'./model/weights/{task}{alg_name}-mat'
        # os.makedirs(output_mat_dir, exist_ok=True)
        #
        # # Loop through each .pth file in the directory
        # for filename in os.listdir(extracted_weights_dir):
        #     if filename.endswith('policy.pth'):
        #         # Load the model weights from the .pth file
        #         model_path = os.path.join(extracted_weights_dir, filename)
        #         model_weights = torch.load(model_path, map_location=torch.device('cpu'))
        #
        #         # Convert the weights to a dictionary format compatible with scipy.io.savemat
        #         weights_dict = {}
        #         for key, value in model_weights.items():
        #             weights_dict[key] = value.cpu().numpy()  # Convert to numpy array for saving
        #
        #         # Define the output path for the .mat file
        #         output_path = os.path.join(output_mat_dir, f"{filename.replace('.pth', '.mat')}")
        #
        #         # Save the weights to a .mat file
        #         sio.savemat(output_path, weights_dict)
        #         print(f"Converted {filename} to {output_path}")




