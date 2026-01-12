"""
This script generates anomaly maps for a given model and category on the MVTec AD dataset.
"""
import os
import shutil
import argparse
import torch
from torchvision import transforms
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh
import numpy as np
import tifffile
from tqdm import tqdm

# --- Configure FiftyOne to use a local database ---
fo.config.database_uri = "mongodb://localhost:27017"

# --- Define Models ---
class DummyModel(torch.nn.Module):
    """A dummy model that returns a random anomaly map."""
    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.image_size = image_size

    def forward(self, image_tensor):
        batch_size = image_tensor.shape[0]
        return torch.rand(batch_size, 1, self.image_size[0], self.image_size[1])

def get_model(model_name, image_size):
    """Factory function to get a model instance by name."""
    if model_name == "DummyModel":
        return DummyModel(image_size=image_size)
    # In the future, add other models here
    # elif model_name == "PatchCore":
    #     return PatchCoreModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate anomaly maps for a given model and category.")
    parser.add_argument('--model_name', required=True, help="Name of the model to use.")
    parser.add_argument('--category', required=True, help="Dataset category to process.")
    parser.add_argument('--output_dir', required=True, help="Root directory to save the anomaly maps.")
    parser.add_argument('--dataset_name', default="mvtec-ad-eval", help="Name for the FiftyOne dataset.")
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help="Image size as a tuple (height, width).")
    return parser.parse_args()

def main():
    """Main function to generate anomaly maps."""
    args = parse_args()
    image_size = tuple(args.image_size)

    print(f"--- Generating maps for model '{args.model_name}' on category '{args.category}' ---")

    # --- Load Dataset via FiftyOne ---
    try:
        if args.dataset_name in fo.list_datasets():
            fo_dataset = fo.load_dataset(args.dataset_name)
        else:
            fo_dataset = fouh.load_from_hub(
                "Voxel51/mvtec-ad",
                dataset_name=args.dataset_name,
                split="test",
            )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Filter dataset for the target category ---
    category_view = fo_dataset.match(F("category.label") == args.category)
    if not category_view:
        print(f"Warning: No samples found for category '{args.category}'. Skipping.")
        return

    # --- Initialize Model ---
    try:
        model = get_model(args.model_name, image_size)
        model.eval()
    except ValueError as e:
        print(f"Error: {e}")
        return

    data_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    
    # --- Generate and Save Anomaly Maps ---
    print(f"Generating anomaly maps and saving to '{args.output_dir}'...")
    with torch.no_grad():
        for sample in tqdm(category_view, desc=f"Processing '{args.category}'"):
            defect_label = sample.defect["label"]
            image_filename = os.path.basename(sample.filepath)
            image_id = os.path.splitext(image_filename)[0]

            image = Image.open(sample.filepath).convert("RGB")
            image_tensor = data_transform(image).unsqueeze(0)
            
            anomaly_map_tensor = model(image_tensor)
            anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()

            # Construct the output path required by the evaluation scripts
            # <base>/<model_name>/<category>/test/<defect_label>/<image_id>.tiff
            map_output_dir = os.path.join(args.output_dir, args.model_name, args.category, "test", defect_label)
            os.makedirs(map_output_dir, exist_ok=True)
            
            output_path = os.path.join(map_output_dir, image_id + ".tiff")
            tifffile.imwrite(output_path, anomaly_map)

    print(f"--- Anomaly map generation for '{args.category}' complete. ---")

if __name__ == "__main__":
    main()
