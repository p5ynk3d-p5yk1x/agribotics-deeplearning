import logging
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from app.model import model

logger = logging.getLogger(__name__)

class InferenceEngine:
    MEAN = (0.485,0.456,0.406)
    STD = (0.229,0.224,0.225)

    def __init__(self,checkpoint_path: str):
        logger.info("Initializing inference engine with checkpoint: %s",checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s",self.device)
        checkpoint = torch.load(checkpoint_path,map_location=self.device)
        logger.info("Checkpoint loaded successfully")
        cfg = checkpoint["cfg"]
        logger.debug("Model config: %s",cfg)
        self.model = model.WeedNet(n_classes=cfg["num_classes"],drop=cfg["dropout"])
        self.model.load_state_dict(checkpoint["state"])
        logger.info("Model weights loaded")
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model moved to %s and set to eval mode",self.device)
        self.classes = [
            name
            for name,_
            in sorted(
                checkpoint["c2i"].items(),
                key=lambda x: x[1]
            )
        ]

        logger.info("Loaded %d classes: %s",len(self.classes),self.classes)
        self.transform = A.Compose([A.Resize(cfg["img_size"],cfg["img_size"]),A.Normalize(self.MEAN,self.STD),ToTensorV2()])
        logger.info("Image transform pipeline initialized (img_size=%s)",cfg["img_size"])

    def predict(self,image: np.ndarray):
        if image is None or not isinstance(image,np.ndarray) or image.size == 0:
            raise ValueError("Inference image is empty or invalid")
        
        logger.info("Running weed inference | Image shape: %s",image.shape)
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb_image)["image"]
        logger.debug("Image transformed | Tensor shape: %s",tuple(tensor.shape))
        tensor = tensor.unsqueeze(0).to(self.device)
        logger.debug("Batch tensor shape: %s",tuple(tensor.shape))

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits,dim=-1)[0]

        confidence,prediction_index = torch.max(probabilities,dim=0)
        weed_name = self.classes[prediction_index.item()]
        confidence_score = float(confidence.item())
        logger.info("Prediction completed | Class: %s | Confidence: %.4f",weed_name,confidence_score)

        return {
            "weedName": weed_name,
            "confidence": confidence_score
        }