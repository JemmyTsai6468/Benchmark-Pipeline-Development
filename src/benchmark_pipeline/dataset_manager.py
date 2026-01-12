# src/benchmark_pipeline/dataset_manager.py
"""
Manages the MVTec AD dataset, including downloading, preparing, and access.
"""
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

from .config import CONFIG

class DatasetManager:
    """
    Handles the setup and access of the MVTec AD dataset.
    """
    def __init__(self, dataset_dir: Path, dataset_name: str = "Voxel51/mvtec-ad", image_size=(256, 256)):
        """
        Initializes the DatasetManager.

        Args:
            dataset_dir: The target directory for the structured dataset components.
            dataset_name: The name for the dataset in FiftyOne.
            image_size: The target size for resizing ground truth masks.
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.image_size = image_size
        self._fo_dataset = None

    def setup_dataset(self):
        """
        Ensures the MVTec AD dataset is downloaded and structured correctly.
        
        Returns:
            The loaded fiftyone.Dataset object.
        """
        print("--- Checking dataset setup ---")
        
        # 1. Load or download the FiftyOne dataset
        self._load_fiftyone_dataset()

        # 2. Restructure dataset into the format required by legacy evaluation scripts
        if not self.dataset_dir.exists() or not any(self.dataset_dir.iterdir()):
             print("Starting one-time dataset restructuring...")
             self._restructure_dataset()
             print("--- Dataset restructuring complete ---\n")
        else:
             print("Structured dataset directory already exists. Skipping.")

        return self._fo_dataset

    def _load_fiftyone_dataset(self):
        """Loads the MVTec AD dataset from the FiftyOne Hub."""
        try:
            if self.dataset_name in fo.list_datasets():
                print(f"Loading existing FiftyOne dataset '{self.dataset_name}'.")
                self._fo_dataset = fo.load_dataset(self.dataset_name)
            else:
                print(f"Downloading dataset from FiftyOne Hub to create '{self.dataset_name}'.")
                self._fo_dataset = fouh.load_from_hub(
                    "Voxel51/mvtec-ad",
                    dataset_name=self.dataset_name,
                )
        except Exception as e:
            print(f"CRITICAL: Error loading dataset from FiftyOne: {e}")
            raise
    
    def _restructure_dataset(self):
        """
        Restructures the FiftyOne dataset into the format required by the
        evaluation scripts.

        The target structure is:
        <dataset_dir>/<category>/test/<defect_type>/<image_name>.png
        <dataset_dir>/<category>/ground_truth/<defect_type>/<image_name>_mask.png
        """
        if self._fo_dataset is None:
            raise ValueError("FiftyOne dataset is not loaded. Cannot restructure.")
            
        print("Restructuring dataset images and ground truth masks...")
        for sample in tqdm(self._fo_dataset, desc="Restructuring dataset"):
            category = sample.category["label"]
            defect_label = sample.defect["label"]
            image_filename = os.path.basename(sample.filepath)
            image_id = os.path.splitext(image_filename)[0]

            # 1. Copy original image to the 'test' directory
            dest_dir = self.dataset_dir / category / "test" / defect_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(sample.filepath, dest_dir / image_filename)

            # 2. Copy and resize ground truth mask if it's an anomaly
            if sample.defect.label != 'good' and sample.defect_mask is not None:
                source_mask_path = Path(sample.defect_mask["mask_path"])
                if source_mask_path.exists():
                    mask = Image.open(source_mask_path)
                    mask = mask.resize(self.image_size, Image.NEAREST)
                    
                    mask_dest_dir = self.dataset_dir / category / "ground_truth" / defect_label
                    mask_dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    mask_filename = f"{image_id}_mask.png"
                    mask.save(mask_dest_dir / mask_filename)
