"""
This script evaluates a model on the MVTec AD dataset.

It uses a dummy model for demonstration purposes. The script performs the
following steps:
1. Loads the MVTec AD dataset using FiftyOne, downloading it locally.
2. Defines a dummy model that generates random anomaly maps.
3. Iterates through a specified category of the dataset (e.g., 'bottle').
4. For each test image, it generates an anomaly map and saves it as a .tiff
   file in the structure required by evaluate_experiment.py.
5. Finally, it runs evaluate_experiment.py to compute performance metrics.
"""

import os
import shutil
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from fiftyone import ViewField as F
import numpy as np
import tifffile
from tqdm import tqdm

# --- Step 1: Configure FiftyOne to use a local database ---
fo.config.database_uri = "mongodb://localhost:27017"

# --- Step 2: Define a Dummy Model ---
class DummyModel(torch.nn.Module):
    """A dummy model that returns a random anomaly map."""
    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.image_size = image_size

    def forward(self, image_tensor):
        """
        Takes a batch of images and returns a batch of random anomaly maps.
        """
        batch_size = image_tensor.shape[0]
        # Generate random anomaly maps of the same size as the input images
        return torch.rand(batch_size, 1, self.image_size[0], self.image_size[1])



def main():
    """
    Main function to run the model evaluation.
    """
    dataset_name = "mvtec-ad-eval"
    structured_dataset_dir = os.path.abspath("./mvtec_ad_structured")
    anomaly_maps_dir = os.path.abspath("./anomaly_maps")
    category_to_test = 'bottle'
    image_size = (256, 256)

    print(f"--- Structured dataset will be stored in: {structured_dataset_dir} ---")
    print(f"--- Anomaly maps will be stored in: {anomaly_maps_dir} ---")

    # --- Step 4: Load Dataset via FiftyOne ---
    try:
        if dataset_name in fo.list_datasets():
            fo_dataset = fo.load_dataset(dataset_name)
            print(f"Loaded existing dataset '{dataset_name}' from FiftyOne DB.")
        else:
            fo_dataset = fouh.load_from_hub(
                "Voxel51/mvtec-ad",
                dataset_name=dataset_name,
                split="test",
            )
            print(f"Dataset loaded from Hub into fiftyone.")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # --- Step 5: Restructure Dataset & Generate Anomaly Maps ---
    print(f"\n--- Restructuring dataset and generating anomaly maps ---")
    
    if os.path.exists(structured_dataset_dir):
        shutil.rmtree(structured_dataset_dir)
    if os.path.exists(anomaly_maps_dir):
        shutil.rmtree(anomaly_maps_dir)
        
    data_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    model = DummyModel(image_size=image_size)
    model.eval()

    with torch.no_grad():
        for sample in tqdm(fo_dataset, desc="Processing samples"):
            category = sample.category["label"]
            defect_label = sample.defect["label"]
            image_filename = os.path.basename(sample.filepath)
            image_id = os.path.splitext(image_filename)[0]
            
            # 1. Restructure original image
            dest_dir = os.path.join(structured_dataset_dir, category, "test", defect_label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, image_filename)
            shutil.copy(sample.filepath, dest_path)

            # 2. Restructure and resize mask
            if sample.defect.label != 'good' and sample.defect_mask is not None:
                source_mask_path = sample.defect_mask["mask_path"]
                if os.path.exists(source_mask_path):
                    mask = Image.open(source_mask_path)
                    mask = mask.resize(image_size, Image.NEAREST)
                    
                    mask_dest_dir = os.path.join(structured_dataset_dir, category, "ground_truth", defect_label)
                    os.makedirs(mask_dest_dir, exist_ok=True)
                    
                    mask_filename = f"{image_id}_mask.png"
                    mask_dest_path = os.path.join(mask_dest_dir, mask_filename)
                    mask.save(mask_dest_path)

            # 3. Generate and save anomaly map (only for the category we test)
            if category == category_to_test:
                image = Image.open(sample.filepath).convert("RGB")
                image_tensor = data_transform(image).unsqueeze(0)
                anomaly_map_tensor = model(image_tensor)
                anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()

                output_dir = os.path.join(anomaly_maps_dir, category, "test", defect_label)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, image_id + ".tiff")
                tifffile.imwrite(output_path, anomaly_map)

    print("\n--- Anomaly map generation and restructuring complete. ---")

    # --- Step 6: Run Evaluation Script ---
    print("\n--- Running evaluation script... ---")
    
    command = [
        "uv", "run", "python", "evaluate_experiment.py",
        "--dataset_base_dir", structured_dataset_dir,
        "--anomaly_maps_dir", anomaly_maps_dir,
        "--output_dir", "./metrics",
        "--evaluated_objects", category_to_test
    ]

    print(f"Executing: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("\n--- Evaluation Script Output ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- Evaluation Script Errors ---")
            print(result.stderr)
        print("---------------------------------")
        print("\nEvaluation finished successfully!")

    except subprocess.CalledProcessError as e:
        print("\n--- Error running evaluation script! ---")
        print(f"Return code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("----------------------------------------")

if __name__ == "__main__":
    main()
