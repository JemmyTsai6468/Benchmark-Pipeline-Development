# src/benchmark_pipeline/generation_runner.py
"""
Handles the process of generating anomaly maps for a given model and dataset category.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as multiprocessing
from torchvision import transforms
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F
import tifffile
from tqdm import tqdm

from .model_handler import ModelHandler
from .config import CONFIG
from .utils import get_device

class FiftyOneDataset(Dataset):
    """
    A PyTorch Dataset to wrap a FiftyOne SampleView, designed to be fork-safe
    for multiprocessing in DataLoader.
    
    The database connection is initialized lazily in each worker process.
    """
    def __init__(self, dataset_name: str, category: str, length: int, transform=None):
        self.dataset_name = dataset_name
        self.category = category
        self.transform = transform
        self._length = length

        # These will be initialized in the worker process
        self.view: fo.SampleView = None
        self.ids = None

    def __len__(self):
        return self._length

    def _init_view(self):
        """Initializes the FiftyOne view within the worker process."""
        # Establish connection and load dataset within the worker
        self.view = fo.load_dataset(self.dataset_name).match(F("category.label") == self.category)
        self.ids = self.view.values("id")

    def __getitem__(self, idx):
        if self.view is None:
            self._init_view()

        sample = self.view[self.ids[idx]]
        image = Image.open(sample.filepath).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        sample_info = {
            'defect_label': sample.defect['label'],
            'filepath': sample.filepath
        }
        
        return image, sample_info

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
        self.model.eval() # Set model to evaluation mode

        self.data_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def run(self):
        """
        Executes the anomaly map generation process.
        """
        print(f"--- Generating maps for model '{self.model_name}' on category '{self.category}' ---")
        
        category_view = self.fo_dataset.match(F("category.label") == self.category)
        view_length = len(category_view)

        if not view_length:
            print(f"Warning: No samples found for category '{self.category}'. Skipping.")
            return

        self._generate_and_save_maps(view_length)
        
        print(f"--- Anomaly map generation for '{self.category}' complete. ---\n")

    def _generate_and_save_maps(self, view_length: int):
        """Iterates through samples, generates anomaly maps, and saves them."""
        
        dataset = FiftyOneDataset(
            dataset_name=self.fo_dataset.name,
            category=self.category,
            length=view_length,
            transform=self.data_transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            multiprocessing_context=multiprocessing.get_context('spawn')
        )

        with torch.no_grad():
            for image_tensor, samples_info in tqdm(dataloader, desc=f"Processing '{self.category}'"):
                image_tensor = image_tensor.to(self.device)
                
                # Generate anomaly map
                anomaly_map_tensor = self.model(image_tensor)
                
                # Process and save each map in the batch
                num_samples = image_tensor.size(0)
                for i in range(num_samples):
                    defect_label = samples_info['defect_label'][i]
                    image_id = Path(samples_info['filepath'][i]).stem
                    
                    anomaly_map = anomaly_map_tensor[i].squeeze().cpu().numpy()

                    # Determine output path and save the map
                    map_output_dir = self.output_dir / self.model_name / self.category / "test" / defect_label
                    map_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = map_output_dir / f"{image_id}.tiff"
                    tifffile.imwrite(output_path, anomaly_map)
