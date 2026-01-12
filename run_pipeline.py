"""
This is the master pipeline script to run a full multi-model, multi-category
evaluation.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from PIL import Image

def run_command(command, description):
    """Runs a command as a subprocess and prints its output."""
    print(f"--- Running: {description} ---")
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"--- Error running '{description}'! ---")
        print(f"Return code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("----------------------------------------")
        raise e
    print(f"--- Finished: {description} ---\n")

def setup_structured_dataset(structured_dataset_dir, dataset_name="mvtec-ad-eval", image_size=(256, 256)):
    """
    Downloads the MVTec AD dataset via FiftyOne and restructures it into the
    format required by the evaluation scripts. This is a one-time setup.
    """
    print("--- Setting up structured dataset (one-time setup) ---")
    if os.path.exists(structured_dataset_dir):
        print(f"'{structured_dataset_dir}' already exists. Skipping setup.")
        return

    os.makedirs(structured_dataset_dir)
    print(f"Created directory '{structured_dataset_dir}'.")

    try:
        if dataset_name in fo.list_datasets():
            fo_dataset = fo.load_dataset(dataset_name)
        else:
            fo_dataset = fouh.load_from_hub(
                "Voxel51/mvtec-ad",
                dataset_name=dataset_name,
            )
    except Exception as e:
        print(f"Error loading dataset from FiftyOne: {e}")
        raise

    print("Restructuring dataset and ground truth masks...")
    for sample in tqdm(fo_dataset, desc="Restructuring dataset"):
        category = sample.category["label"]
        defect_label = sample.defect["label"]
        image_filename = os.path.basename(sample.filepath)
        image_id = os.path.splitext(image_filename)[0]

        # Copy original image
        dest_dir = os.path.join(structured_dataset_dir, category, "test", defect_label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(sample.filepath, os.path.join(dest_dir, image_filename))

        # Copy and resize mask
        if sample.defect.label != 'good' and sample.defect_mask is not None:
            source_mask_path = sample.defect_mask["mask_path"]
            if os.path.exists(source_mask_path):
                mask = Image.open(source_mask_path)
                mask = mask.resize(image_size, Image.NEAREST)
                mask_dest_dir = os.path.join(structured_dataset_dir, category, "ground_truth", defect_label)
                os.makedirs(mask_dest_dir, exist_ok=True)
                mask_filename = f"{image_id}_mask.png"
                mask.save(os.path.join(mask_dest_dir, mask_filename))
    
    print("--- Dataset setup complete ---\n")


def main():
    """Main function to orchestrate the evaluation pipeline."""
    # 1. Load Configuration
    try:
        with open("pipeline_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: pipeline_config.json not found. Please create it.")
        return

    models = config["models"]
    categories = config["categories"]
    dataset_dir = config["dataset_dir"]
    anomaly_maps_root = config["anomaly_maps_root"]
    metrics_root = config["metrics_root"]
    eval_config_path = "eval_config.json"

    # 2. Prepare Directories
    print("--- Preparing directories ---")
    if os.path.exists(anomaly_maps_root):
        shutil.rmtree(anomaly_maps_root)
    os.makedirs(anomaly_maps_root)
    
    if os.path.exists(metrics_root):
        shutil.rmtree(metrics_root)
    os.makedirs(metrics_root)
    print(f"Cleaned and created '{anomaly_maps_root}' and '{metrics_root}'.\n")

    # 3. One-Time Dataset Setup
    setup_structured_dataset(dataset_dir)

    # 4. Loop and Generate Anomaly Maps
    print("--- Starting anomaly map generation phase ---")
    for model_name in models:
        for category in categories:
            command = [
                "uv", "run", "run_generation.py",
                "--model_name", model_name,
                "--category", category,
                "--output_dir", anomaly_maps_root,
            ]
            run_command(command, f"Generate maps for {model_name} on {category}")
    print("--- Anomaly map generation complete ---\n")

    # 5. Create Evaluation Config
    print("--- Creating config for multi-experiment evaluation ---")
    anomaly_maps_dirs = {}
    for model_name in models:
        # The evaluation script needs paths relative to the exp_base_dir
        anomaly_maps_dirs[model_name] = f"{model_name}/"
    
    eval_config = {
        "exp_base_dir": str(Path(anomaly_maps_root).resolve()),
        "anomaly_maps_dirs": anomaly_maps_dirs
    }

    with open(eval_config_path, "w") as f:
        json.dump(eval_config, f, indent=4)
    print(f"Created '{eval_config_path}'.\n")

    # 6. Run Multi-Experiment Evaluation
    command = [
        "uv", "run", "evaluate_multiple_experiments.py",
        "--dataset_base_dir", dataset_dir,
        "--experiment_configs", eval_config_path,
        "--output_dir", metrics_root
    ]
    run_command(command, "Run multi-experiment evaluation")

    # 7. Print Final Results
    command = [
        "uv", "run", "print_metrics.py",
        "--metrics_folder", metrics_root
    ]
    run_command(command, "Print summary of results")
    
    print("====== Pipeline Finished Successfully! ======")

if __name__ == "__main__":
    main()