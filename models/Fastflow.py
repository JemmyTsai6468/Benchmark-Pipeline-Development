# models/Fastflow.py
import torch
from torch.utils.data import DataLoader

from anomalib.models.image.fastflow.torch_model import FastflowModel as AnomalibFastflowModel
from anomalib.models.image.fastflow.loss import FastflowLoss
from src.benchmark_pipeline.model_handler import BenchmarkModel

class Fastflow(BenchmarkModel):
    def __init__(self, image_size=(256, 256)):
        """
        Initializes the Fastflow model wrapper.
        """
        super().__init__(image_size)
        
        self.model = AnomalibFastflowModel(
            input_size=image_size,
            backbone='resnet18',
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
            pre_trained=True,
        )
        
        self.loss = FastflowLoss()

    def fit(self, training_dataloader: DataLoader):
        """
        Trains the Fastflow model using the logic from the anomalib lightning model.
        """
        print(f"[{self.__class__.__name__}] Starting fitting process...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.00001
        )
        
        self.model.train()
        
        # For simplicity, we train for a fixed number of epochs.
        num_epochs = 2
        for epoch in range(num_epochs):
            print(f"[{self.__class__.__name__}] Epoch {epoch + 1}/{num_epochs}")
            total_loss = 0.0
            for batch in training_dataloader:
                optimizer.zero_grad()
                
                images = batch[0].to(device)
                
                hidden_variables, log_jacobians = self.model(images)
                
                loss = self.loss(hidden_variables, log_jacobians)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[{self.__class__.__name__}] Average Loss: {total_loss / len(training_dataloader)}")

        print(f"[{self.__class__.__name__}] Fitting complete.")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs inference using the trained Fastflow model.
        """
        self.model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_tensor = image_tensor.to(device)
        
        output = self.model(image_tensor)
        
        return output.anomaly_map
