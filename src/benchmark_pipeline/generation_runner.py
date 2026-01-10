# src/benchmark_pipeline/generation_runner.py
"""
Handles the process of generating anomaly maps for a given model and dataset category.
"""
import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F
import tifffile
from tqdm import tqdm

from .model_handler import ModelHandler
from .config import CONFIG

class GenerationRunner:
    """
    Orchestrates the anomaly map generation for a single model-category pair.
    """
    def __init__(self, model_name: str, category: str, fo_dataset: 'fo.Dataset', image_size=(256, 256)):
        """
        Initializes the GenerationRunner.

        Args:
            model_name: The name of the model to use.
            category: The dataset category to process.
            fo_dataset: The loaded fiftyone dataset object.
            image_size: The target image size.
        """
        self.model_name = model_name
        self.category = category
        self.fo_dataset = fo_dataset
        self.image_size = image_size
        self.output_dir = CONFIG.paths.anomaly_maps_root
        
        # Initialize the model using the handler
        handler = ModelHandler(model_name)
        self.model = handler.get_model_instance(image_size=self.image_size)
        
        self.data_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # Add normalization if required by models, but keep it simple for now
        ])

    def run(self):
        """
        Executes the anomaly map generation process.
        """
        print(f"--- Generating maps for model '{self.model_name}' on category '{self.category}' ---")
        
        category_view = self._get_dataset_view()
        if not category_view:
            print(f"Warning: No samples found for category '{self.category}'. Skipping.")
            return

        self._generate_and_save_maps(category_view)
        
        print(f"--- Anomaly map generation for '{self.category}' complete. ---\n")

    def _get_dataset_view(self):
        """Filters the FiftyOne dataset to get the view for the target category."""
        try:
            return self.fo_dataset.match(F("category.label") == self.category)
        except Exception as e:
            print(f"Error filtering dataset view: {e}")
            raise

    def _generate_and_save_maps(self, category_view):
        """Iterates through samples, generates anomaly maps, and saves them."""
        with torch.no_grad():
            for sample in tqdm(category_view, desc=f"Processing '{self.category}'"):
                defect_label = sample.defect["label"]
                image_id = Path(sample.filepath).stem

                # Prepare image
                image = Image.open(sample.filepath).convert("RGB")
                image_tensor = self.data_transform(image).unsqueeze(0)
                
                # Generate anomaly map
                anomaly_map_tensor = self.model(image_tensor)
                anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()

                # Determine output path and save the map
                map_output_dir = self.output_dir / self.model_name / self.category / "test" / defect_label
                map_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = map_output_dir / f"{image_id}.tiff"
                tifffile.imwrite(output_path, anomaly_map)
