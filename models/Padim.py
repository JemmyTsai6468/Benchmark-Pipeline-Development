# models/Padim.py
import torch
from torch.utils.data import DataLoader

# Import the correct PadimModel as specified by the user
from anomalib.models.image.padim.torch_model import PadimModel as AnomalibPadim
from src.benchmark_pipeline.model_handler import BenchmarkModel

class Padim(BenchmarkModel):
    def __init__(self, image_size=(256, 256)):
        """
        Initializes the Padim model wrapper.
        """
        super().__init__(image_size)
        
        # 1. Instantiate the Padim model from anomalib
        # We use the specific torch_model as requested.
        self.model = AnomalibPadim(
            backbone='resnet18',
            layers=['layer1', 'layer2', 'layer3'],
            pre_trained=True,
            n_features=100  # Default for resnet18
        )

    def fit(self, training_dataloader: DataLoader):
        """
        Trains the Padim model by collecting embeddings and fitting a Gaussian.
        """
        print(f"[{self.__class__.__name__}] Starting fitting process...")
        
        # 1. Set model to training mode
        self.model.train()
        
        # Move model to the correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        # 2. Iterate through the dataloader to collect embeddings
        with torch.no_grad(): # Disable gradient computation during feature extraction
            for batch in training_dataloader:
                images = batch[0].to(device)
                # The forward pass in training mode populates the memory bank
                self.model(images)
        
        # 3. Fit the Gaussian model
        print(f"[{self.__class__.__name__}] Fitting Gaussian model...")
        self.model.fit()
        print(f"[{self.__class__.__name__}] Fitting complete.")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs inference using the trained Padim model.
        """
        # 1. Set model to evaluation mode
        self.model.eval()
        
        # 2. Perform inference
        # The forward pass in eval mode returns an InferenceBatch
        output = self.model(image_tensor)
        
        # 3. Extract and return the anomaly map
        # The output object has an 'anomaly_map' attribute
        return output.anomaly_map
