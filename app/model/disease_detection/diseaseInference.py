import json
import logging
import os
from typing import Optional
import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

PLANT_DISEASES = {
    "Apple": ["Apple_Scab","Black_Rot","Cedar_Apple_Rust","Healthy","Common_Rust","Scab_Leaf"],
    "Banana": ["Healthy","Bract_Mosaic_Virus","Cordana","Insect_Pest","Moko","Panama_Disease","Pestalotiopsis","Sigatoka","Yellow_Sigatoka"],
    "Bell": ["Bell_Bacterial_Spot","Bell_Healthy","Pepper","Pepper_Spot"],
    "Blueberry": ["Healthy"],
    "Cauliflower": ["Healthy","Black_Rot","Bacterial_Spot_Rot","Downy_Mildew"],
    "Cherry": ["Healthy","Powdery_Mildew"],
    "Chilli": ["Healthy","Anthracnose","Leaf_Curl","Leaf_Spot","Whitefly","Yellowing"],
    "Corn": ["(Maize)_Healthy","(Maize)_Common_Rust","(Maize)_Northern_Leaf_Blight","(Maize)_Cercospora_Leaf_Spot_Gray_Leaf_Spot","Blight","Gray_Spot","Bacterial_leaf_blight","Brown_spot","Gray_Leaf_Spot","Leaf_smut"],
    "Grape": ["Healthy","Black_Rot","Esca_(Black_Measles)","Leaf_Blight_(Isariopsis_Leaf_Spot)","(Including_Sour)_Healthy","(Including_Sour)_Powdery_Mildew"],
    "Groundnut": ["Healthy","Early_Leaf_Spot","Early_Rust","Late_Leaf_Spot","Nutrition_Deficiency","Common_Rust"],
    "Orange": ["Haunglongbing_(Citrus_Greening)"],
    "Peach": ["Healthy","Bacterial_Spot"],
    "Pepper": ["Bell_Bacterial_Spot","Bell_Healthy","Pepper","Pepper_Spot"],
    "Potato": ["Healthy","Early_Blight","Late_Blight"],
    "Radish": ["Healthy","Black_Leaf_Spot","Downy_Mildew","Flea_Beetle","Mosaic"],
    "Raspberry": ["Healthy"],
    "Soybean": ["Healthy"],
    "Soyabean": ["Healthy"],
    "Squash": ["Powdery_Mildew"],
    "Strawberry": ["Healthy","Leaf_Scorch"],
    "Tomato": ["Healthy","Bacterial_Spot","Early_Blight","Late_Blight","Leaf_Mold","Mosaic_Virus","Septoria_Leaf_Spot","Septoria_Spot","Target_Spot","Tomato_Mosaic_Virus","Tomato_Yellow_Leaf_Curl_Virus","Two_Spotted_Spider_Mites","Spider_Mites_Two-Spotted_Spider_Mite","Yellow_Leaf_Curl_Virus"],
    "grape": ["Healthy","Black_Rot","Esca_(Black_Measles)","Leaf_Blight_(Isariopsis_Leaf_Spot)","(Including_Sour)_Healthy","(Including_Sour)_Powdery_Mildew"]
}

class ConvNeXtV2Classifier(nn.Module):
    def __init__(self,num_classes: int):
        super().__init__()
        self.model = timm.create_model("convnextv2_tiny",pretrained=False,num_classes=num_classes)

    def forward(self,image: torch.Tensor):
        return self.model(image)

class DiseaseInferenceEngine:
    MEAN = (0.485,0.456,0.406)
    STD = (0.229,0.224,0.225)

    def __init__(self,plant_model_path: str,disease_model_path: str,plant_mapping_path: str,disease_mapping_path: str,image_size: int = 224,plant_confidence_threshold: float = 0.80,top_k: int = 5):
        logger.info("Initializing DiseaseInferenceEngine")
        self._validate_file(plant_model_path,"plant model")
        self._validate_file(disease_model_path,"disease model")
        self._validate_file(plant_mapping_path,"plant class mapping")
        self._validate_file(disease_mapping_path,"disease class mapping")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.plant_confidence_threshold = plant_confidence_threshold
        self.top_k = top_k
        logger.info("Disease inference device selected | Device: %s",self.device)
        self.plant_names = self._load_class_mapping(plant_mapping_path,"plant")
        self.disease_names = self._load_class_mapping(disease_mapping_path,"disease")
        self.plant_model = ConvNeXtV2Classifier(len(self.plant_names))
        self.disease_model = ConvNeXtV2Classifier(len(self.disease_names))
        self._load_model_weights(self.plant_model,plant_model_path,"plant")
        self._load_model_weights(self.disease_model,disease_model_path,"disease")
        self.transform = A.Compose([
            A.Resize(self.image_size,self.image_size),
            A.Normalize(mean=self.MEAN,std=self.STD),
            ToTensorV2()
        ])
        self._validate_model_output(self.plant_model,len(self.plant_names),"plant")
        self._validate_model_output(self.disease_model,len(self.disease_names),"disease")
        logger.info("DiseaseInferenceEngine initialized successfully | Architecture: convnextv2_tiny | Plant classes: %d | Disease classes: %d | Image size: %d | Threshold: %.2f | Top K: %d",len(self.plant_names),len(self.disease_names),self.image_size,self.plant_confidence_threshold,self.top_k)

    @staticmethod
    def _validate_file(path: str,description: str):
        if not os.path.isfile(path):
            logger.error("Required file not found | Type: %s | Path: %s",description,path)
            raise FileNotFoundError(f"{description.capitalize()} not found: {path}")
        logger.info("Required file found | Type: %s | Path: %s",description,path)

    @staticmethod
    def _load_class_mapping(mapping_path: str,mapping_type: str):
        logger.info("Loading class mapping | Type: %s | Path: %s",mapping_type,mapping_path)
        with open(mapping_path,"r",encoding="utf-8") as mapping_file:
            mapping = json.load(mapping_file)
        classes = mapping.get("classes")
        declared_count = mapping.get("num_classes")
        if not isinstance(classes,list) or not classes:
            logger.error("Invalid class mapping | Type: %s | Missing non-empty classes list",mapping_type)
            raise ValueError(f"Invalid {mapping_type} class mapping")
        if declared_count is not None and declared_count != len(classes):
            logger.error("Class mapping count mismatch | Type: %s | Declared: %s | Actual: %d",mapping_type,declared_count,len(classes))
            raise ValueError(f"{mapping_type.capitalize()} class mapping count mismatch")
        logger.info("Class mapping loaded | Type: %s | Classes: %d",mapping_type,len(classes))
        return classes

    @staticmethod
    def _extract_state_dict(checkpoint,model_type: str):
        if not isinstance(checkpoint,dict):
            logger.error("Invalid checkpoint type | Type: %s | Checkpoint type: %s",model_type,type(checkpoint))
            raise TypeError(f"Invalid {model_type} checkpoint type: {type(checkpoint)}")
        for key in ("state_dict","model_state_dict","state"):
            candidate = checkpoint.get(key)
            if isinstance(candidate,dict):
                logger.info("State dictionary extracted from checkpoint | Type: %s | Key: %s",model_type,key)
                return candidate
        return checkpoint

    def _load_model_weights(self,model: nn.Module,model_path: str,model_type: str):
        logger.info("Loading ConvNeXt V2 Tiny weights | Type: %s | Path: %s",model_type,model_path)
        try:
            checkpoint = torch.load(model_path,map_location=self.device,weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path,map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint,model_type)
        normalized_state = {}
        for key,value in state_dict.items():
            normalized_key = key.removeprefix("module.")
            normalized_state[normalized_key] = value
        try:
            model.load_state_dict(normalized_state,strict=True)
        except RuntimeError:
            logger.exception("Model weights do not match ConvNeXt V2 Tiny | Type: %s | Path: %s",model_type,model_path)
            raise
        model.to(self.device)
        model.eval()
        logger.info("ConvNeXt V2 Tiny weights loaded successfully | Type: %s | Parameters: %d",model_type,sum(parameter.numel() for parameter in model.parameters()))

    def _validate_model_output(self,model: nn.Module,expected_classes: int,model_type: str):
        logger.info("Validating model output | Type: %s | Expected classes: %d",model_type,expected_classes)
        sample = torch.zeros(1,3,self.image_size,self.image_size,device=self.device)
        with torch.inference_mode():
            output = model(sample)
        if not isinstance(output,torch.Tensor):
            logger.error("Model returned an unsupported output | Type: %s | Output type: %s",model_type,type(output))
            raise TypeError(f"Unsupported {model_type} model output: {type(output)}")
        if output.ndim != 2 or output.shape != (1,expected_classes):
            logger.error("Unexpected model output shape | Type: %s | Expected: %s | Actual: %s",model_type,(1,expected_classes),tuple(output.shape))
            raise ValueError(f"Unexpected {model_type} model output shape: {tuple(output.shape)}")
        logger.info("Model output validated | Type: %s | Shape: %s",model_type,tuple(output.shape))

    def _prepare_image(self,image: np.ndarray,job_id: str):
        if image is None or not isinstance(image,np.ndarray) or image.size == 0:
            raise ValueError(f"Disease image is empty or invalid for job: {job_id}")
        logger.info("Preparing disease image | Job ID: %s | Shape: %s",job_id,image.shape)
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb_image)["image"].unsqueeze(0).to(self.device)
        logger.info("Disease image transformed | Job ID: %s | Tensor shape: %s",job_id,tuple(tensor.shape))
        return tensor

    def _predict_plant(self,tensor: torch.Tensor,job_id: str):
        logger.info("Starting plant classification | Job ID: %s",job_id)
        with torch.inference_mode():
            probabilities = torch.softmax(self.plant_model(tensor),dim=1)
            confidence,index = torch.max(probabilities,dim=1)
        plant_name = self.plant_names[index.item()]
        plant_confidence = float(confidence.item())
        logger.info("Plant classification completed | Job ID: %s | Plant: %s | Confidence: %.4f",job_id,plant_name,plant_confidence)
        return plant_name,plant_confidence

    def _predict_disease(self,tensor: torch.Tensor,plant_name: str,job_id: str):
        logger.info("Starting disease classification | Job ID: %s | Plant: %s",job_id,plant_name)
        with torch.inference_mode():
            probabilities = torch.softmax(self.disease_model(tensor),dim=1)[0]
        allowed_diseases = PLANT_DISEASES.get(plant_name,self.disease_names)
        allowed_indices = [
            self.disease_names.index(name)
            for name in allowed_diseases
            if name in self.disease_names
        ]
        if allowed_indices:
            mask = torch.zeros(len(self.disease_names),device=self.device)
            mask[allowed_indices] = 1
            filtered = probabilities * mask
            filtered_sum = float(filtered.sum().item())
            if filtered_sum > 0:
                filtered = filtered / filtered_sum
                logger.info("Plant-specific disease filter applied | Job ID: %s | Plant: %s | Allowed classes: %d",job_id,plant_name,len(allowed_indices))
            else:
                filtered = probabilities
                logger.warning("Disease filter produced an empty distribution; using original probabilities | Job ID: %s",job_id)
        else:
            filtered = probabilities
            logger.warning("No matching disease classes configured; using all disease classes | Job ID: %s | Plant: %s",job_id,plant_name)
        result_count = min(self.top_k,len(self.disease_names))
        top_probabilities,top_indices = torch.topk(filtered,result_count)
        predictions = [
            {
                "rank": rank,
                "diseaseName": self.disease_names[index.item()],
                "confidence": float(probability.item())
            }
            for rank,(index,probability) in enumerate(zip(top_indices,top_probabilities),start=1)
        ]
        for prediction in predictions:
            logger.info("Disease candidate | Job ID: %s | Rank: %d | Disease: %s | Confidence: %.4f",job_id,prediction["rank"],prediction["diseaseName"],prediction["confidence"])
        logger.info("Disease classification completed | Job ID: %s | Disease: %s | Confidence: %.4f",job_id,predictions[0]["diseaseName"],predictions[0]["confidence"])
        return predictions

    def _create_output_image(self,image: np.ndarray,job_id: str,plant_name: str,plant_confidence: float,disease_name: Optional[str],disease_confidence: Optional[float],skipped: bool) -> np.ndarray:
        logger.info("Creating disease output image in memory | Job ID: %s",job_id)
        panel_height = 150 if skipped else 190
        annotated = cv2.copyMakeBorder(image,panel_height,0,0,0,cv2.BORDER_CONSTANT,value=(28,28,28))
        cv2.putText(annotated,f"Plant: {plant_name}",(20,38),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(annotated,f"Plant confidence: {plant_confidence * 100:.2f}%",(20,78),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2,cv2.LINE_AA)
        if skipped:
            cv2.putText(annotated,"Disease prediction skipped",(20,118),cv2.FONT_HERSHEY_SIMPLEX,0.68,(0,190,255),2,cv2.LINE_AA)
        else:
            cv2.putText(annotated,f"Disease: {disease_name}",(20,118),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(annotated,f"Disease confidence: {disease_confidence * 100:.2f}%",(20,158),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2,cv2.LINE_AA)
        logger.info("Disease output image created in memory | Job ID: %s | Shape: %s",job_id,annotated.shape)
        return annotated

    def predict(self,image: np.ndarray,job_id: str):
        logger.info("Starting disease prediction | Job ID: %s",job_id)
        try:
            tensor = self._prepare_image(image,job_id)
            plant_name,plant_confidence = self._predict_plant(tensor,job_id)
            if plant_confidence < self.plant_confidence_threshold:
                logger.warning("Plant confidence below threshold; disease prediction skipped | Job ID: %s | Plant: %s | Confidence: %.4f | Threshold: %.4f",job_id,plant_name,plant_confidence,self.plant_confidence_threshold)
                annotated_image = self._create_output_image(image,job_id,plant_name,plant_confidence,None,None,True)
                result = {
                    "plantName": plant_name,
                    "plantConfidence": plant_confidence,
                    "diseaseName": None,
                    "diseaseConfidence": None,
                    "diseasePredictionSkipped": True,
                    "skipReason": "PLANT_CONFIDENCE_BELOW_THRESHOLD",
                    "topPredictions": []
                }
                logger.info("Disease prediction completed with plant-confidence rejection | Job ID: %s",job_id)
                return result,annotated_image
            predictions = self._predict_disease(tensor,plant_name,job_id)
            primary_prediction = predictions[0]
            annotated_image = self._create_output_image(
                image,
                job_id,
                plant_name,
                plant_confidence,
                primary_prediction["diseaseName"],
                primary_prediction["confidence"],
                False
            )
            result = {
                "plantName": plant_name,
                "plantConfidence": plant_confidence,
                "diseaseName": primary_prediction["diseaseName"],
                "diseaseConfidence": primary_prediction["confidence"],
                "diseasePredictionSkipped": False,
                "skipReason": None,
                "topPredictions": predictions
            }
            logger.info("Disease prediction completed successfully | Job ID: %s | Plant: %s | Disease: %s",job_id,plant_name,primary_prediction["diseaseName"])
            return result,annotated_image
        except Exception:
            logger.exception("Disease prediction failed | Job ID: %s",job_id)
            raise
