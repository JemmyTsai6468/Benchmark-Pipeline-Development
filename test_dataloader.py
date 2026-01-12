import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import fiftyone as fo
import fiftyone.utils.huggingface as fouh
import numpy as np

# --- Step 1: Directly configure FiftyOne to use the Dockerized MongoDB ---
fo.config.database_uri = "mongodb://localhost:27017"


class MVTecADFiftyOneDataset(Dataset):
    """Custom PyTorch dataset for the MVTec AD dataset loaded via FiftyOne."""

    def __init__(self, fiftyone_dataset, transform=None, target_transform=None):
        self.samples = fiftyone_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.ids = self.samples.values("id")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample = self.samples[sample_id]

        image = Image.open(sample.filepath).convert("RGB")
        img_width, img_height = image.size # Get dimensions from the actual image
        
        # Determine if abnormal based on 'defect.label'
        is_abnormal = 1 if sample.defect.label != 'good' else 0
        
        if is_abnormal and sample.defect_mask is not None and sample.defect_mask.mask is not None:
            # For ImageSegmentation field, the mask is directly in its 'mask' attribute
            mask_np = sample.defect_mask.mask
            mask = torch.from_numpy(mask_np).float().unsqueeze(0) # Add channel dimension
        else:
            # Create an empty mask for normal images or if defect_mask is None or mask_np is None
            mask = torch.zeros(1, img_height, img_width)
            
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask, torch.tensor(is_abnormal, dtype=torch.long)

def main():
    print("--- Loading dataset via FiftyOne (with Docker backend) ---")
    dataset_name = "mvtec-ad-test"
    
    # Clean up previous dataset from fiftyone DB if it exists
    # Temporarily commented out for debugging
    # if dataset_name in fo.list_datasets():
    #     fo.delete_dataset(dataset_name)

    try:
        # Check if dataset already exists in fiftyone DB
        if dataset_name in fo.list_datasets():
            fo_dataset = fo.load_dataset(dataset_name)
            print(f"Loaded existing dataset '{dataset_name}' from FiftyOne database.")
        else:
            # Load the 'test' split. The loader can filter by class.
            fo_dataset = fouh.load_from_hub(
                "Voxel51/mvtec-ad",
                dataset_name=dataset_name,
                split="test",
            )
            print(f"\nDataset loaded successfully via FiftyOne! Found {len(fo_dataset)} total test samples.")

        # --- Debugging: Print first sample to understand structure ---
        print("\n--- First Sample Debug Info ---")
        first_sample = fo_dataset.first()
        print(first_sample)
        print("-------------------------------")
        # --- End Debugging ---

        # Filter for the 'bottle' category and 'test' split
        category_to_test = 'bottle'
        print(f"\nFiltering for category: '{category_to_test}' and split 'test'...")
        
        # Filtering by category.label and split field
        # The Voxel51/mvtec-ad dataset combines all splits and categories,
        # so we need to filter them out explicitly.
        category_view = fo_dataset.match(F("category.label") == category_to_test).match(F("split") == "test")
        
        if len(category_view) == 0:
            print(f"Error: No samples found for category '{category_to_test}'. The filter might be incorrect or category tag is missing.")
            return

        print(f"Found {len(category_view)} samples for category '{category_to_test}'.")

        # Define transformations
        image_size = (256, 256)
        data_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        target_transform = transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)])

        print("\nCreating PyTorch Dataset for 'bottle' category...")
        # The view is passed to the torch dataset
        mvtec_dataset = MVTecADFiftyOneDataset(
            fiftyone_dataset=category_view,
            transform=data_transform,
            target_transform=target_transform
        )
        print(f"Custom dataset created with {len(mvtec_dataset)} samples.")

        dataloader = DataLoader(mvtec_dataset, batch_size=4, shuffle=True)

        print("\nTesting DataLoader...")
        images, masks, is_abnormal = next(iter(dataloader))
        
        print(f"Successfully loaded one batch of data.")
        print(f"Images batch shape: {images.shape}")
        print(f"Masks batch shape: {masks.shape}")
        print(f"Is Abnormal batch: {is_abnormal}")
        
        print("\nTest finished successfully! The DataLoader is working correctly.")
        
    except StopIteration:
        print("DataLoader is empty.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    # finally: # This finally block was commented out for debugging.
        # Clean up the dataset record from the fiftyone database
        # Commented out for debugging, uncomment when working
        # if dataset_name in fo.list_datasets():
        #     print(f"\nDeleting '{dataset_name}' record from FiftyOne database...")
        #     fo.delete_dataset(dataset_name)
        #     print("Dataset record deleted.")


if __name__ == "__main__":
    # fiftyone's filter requires this import
    from fiftyone import ViewField as F
    main()
