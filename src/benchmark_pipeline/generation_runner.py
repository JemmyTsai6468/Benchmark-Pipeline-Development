# src/benchmark_pipeline/generation_runner.py
"""
Handles the process of generating anomaly maps for a given model and dataset category.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torchvision import transforms
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.core.sample import SampleView
from fiftyone.utils.torch import FiftyOneTorchDataset, GetItem
import tifffile
from tqdm import tqdm

from .model_handler import ModelHandler
from .config import CONFIG
from .utils import get_device

# Define the transformation pipeline
DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class AnomalyMapGetItem(GetItem):
    """
    A custom GetItem class for the vectorized FiftyOneTorchDataset.
    It defines the fields to be cached and the logic to transform them.
    """
    @property
    def required_keys(self):
        # Define the fields to cache from the FiftyOne dataset
        return "filepath", "defect"

    def __call__(self, sample_dict):
        """
        Loads and transforms the cached data for a single sample.
        
        Args:
            sample_dict: A dict containing the cached field values.
        
        Returns:
            A tuple of (image_tensor, defect_label, image_id)
        """
        filepath = sample_dict["filepath"]
        defect_label = sample_dict["defect"]["label"]
        image_id = Path(filepath).stem
        
        image = Image.open(filepath).convert("RGB")
        image_tensor = DATA_TRANSFORM(image)
        
        return image_tensor, defect_label, image_id

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
        self.batch_size = CONFIG.batch_size

        # Determine device and move model
        self.device = get_device(CONFIG.device)
        print(f"INFO: Using device: {self.device}")
        
        handler = ModelHandler(model_name)
        self.model = handler.get_model_instance(image_size=self.image_size)
        self.model.to(self.device)
        self.model.eval()

    def run(self):
        """
        Executes the anomaly map generation process.
        """
        print(f"--- Generating maps for model '{self.model_name}' on category '{self.category}' ---")
        
        category_view = self.fo_dataset.match(F("category.label") == self.category)

        if not category_view:
            print(f"Warning: No samples found for category '{self.category}'. Skipping.")
            return

        self._generate_and_save_maps(category_view)
        
        print(f"--- Anomaly map generation for '{self.category}' complete. ---\n")

    def _generate_and_save_maps(self, category_view: SampleView):
        """Iterates through samples, generates anomaly maps, and saves them."""
        
        # Create a vectorized dataset for performance
        torch_dataset = FiftyOneTorchDataset(
            category_view,
            get_item=AnomalyMapGetItem(),
            vectorize=True,
        )

        dataloader = DataLoader(
            torch_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            for image_tensor, defect_labels, image_ids in tqdm(dataloader, desc=f"Processing '{self.category}'"):
                image_tensor = image_tensor.to(self.device)
                
                # Generate anomaly map
                anomaly_map_tensor = self.model(image_tensor)
                
                # Process and save each map in the batch
                for i in range(len(image_ids)):
                    defect_label = defect_labels[i]
                    image_id = image_ids[i]
                    
                    anomaly_map = anomaly_map_tensor[i].squeeze().cpu().numpy()

                    # Determine output path and save the map
                    map_output_dir = self.output_dir / self.model_name / self.category / "test" / defect_label
                    map_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = map_output_dir / f"{image_id}.tiff"
                    tifffile.imwrite(output_path, anomaly_map)
