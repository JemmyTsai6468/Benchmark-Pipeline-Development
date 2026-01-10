# models/PatchCore.py
import torch
import timm
from torch.utils.data import DataLoader
from src.benchmark_pipeline.model_handler import BenchmarkModel
import torch.nn.functional as F

class PatchCore(BenchmarkModel):
    def __init__(self, image_size=(256, 256), coreset_sampling_ratio: float = 0.01):
        super().__init__(image_size)
        
        self.coreset_sampling_ratio = coreset_sampling_ratio

        # 1. Load pre-trained backbone (e.g., Wide-Res-Net-50)
        # `features_only=True` returns intermediate layer features
        print(f"[{self.__class__.__name__}] Loading backbone: wide_resnet50_2")
        self.backbone = timm.create_model(
            'wide_resnet50_2',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3] # Extract features from intermediate layers 2 and 3
        ).eval() # Set to evaluation mode

        self.memory_bank = [] # Initialize memory bank

    def fit(self, training_dataloader: DataLoader):
        """Build the feature memory bank using normal training data."""
        print(f"[{self.__class__.__name__}] Building memory bank...")
        all_features = []
        with torch.no_grad():
            for samples in training_dataloader:
                images = samples[0] # Adjust based on your DataLoader
                if torch.cuda.is_available():
                    images = images.cuda()

                # Extract features
                # Note: This is a simplified feature extraction. A real implementation
                # would involve more sophisticated patch-level processing.
                features = self.backbone(images)
                
                # For this placeholder, we'll just use the flattened output of the last feature map
                layer_2_features = F.avg_pool2d(features[0], 3, 1, 1)
                layer_3_features = F.avg_pool2d(features[1], 3, 1, 1)
                
                # This is a simplified feature combination
                combined_features = torch.cat([layer_2_features, F.interpolate(layer_3_features, scale_factor=2)], dim=1)
                
                all_features.append(combined_features.cpu().view(combined_features.shape[0], combined_features.shape[1], -1).permute(0, 2, 1).reshape(-1, combined_features.shape[1]))

        # In a real implementation, you would perform coreset subsampling here
        self.memory_bank = torch.cat(all_features, dim=0)

        # Coreset Subsampling
        if self.coreset_sampling_ratio < 1.0:
            num_features_to_keep = int(self.memory_bank.shape[0] * self.coreset_sampling_ratio)
            perm = torch.randperm(self.memory_bank.shape[0])
            self.memory_bank = self.memory_bank[perm[:num_features_to_keep]]

        if torch.cuda.is_available():
            self.memory_bank = self.memory_bank.cuda()

        print(f"[{self.__class__.__name__}] Memory bank built with {self.memory_bank.shape[0]} features.")
        
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Inference phase: calculate anomaly map."""
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("The model has not been fitted. Call .fit() before forward.")

        with torch.no_grad():
            # 1. Extract features from the test image
            features = self.backbone(image_tensor)
            layer_2_features = F.avg_pool2d(features[0], 3, 1, 1)
            layer_3_features = F.avg_pool2d(features[1], 3, 1, 1)
            test_features = torch.cat([layer_2_features, F.interpolate(layer_3_features, scale_factor=2)], dim=1)
            
            # 2. Calculate distances to the memory bank (nearest neighbor)
            # This is a simplified, non-optimized distance calculation
            test_patches = test_features.view(test_features.shape[0], test_features.shape[1], -1).permute(0, 2, 1).reshape(-1, test_features.shape[1])
            
            # Using torch.cdist for batch distance calculation
            distances = torch.cdist(test_patches, self.memory_bank).min(dim=1)[0]
            
            # 3. Reshape distances into an anomaly map
            map_size = test_features.shape[2] # Use the actual feature map size
            anomaly_map = distances.view(image_tensor.shape[0], map_size, map_size).unsqueeze(1)
            
            # 4. Upsample the anomaly map to the original image size
            anomaly_map_resized = F.interpolate(
                anomaly_map,
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            )

        return anomaly_map_resized
