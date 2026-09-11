import os

AWS_REGION = os.getenv("AWS_REGION","ap-south-1")
AWS_S3_UPLOAD_BUCKET = os.getenv("AWS_S3_UPLOAD_BUCKET","uploads-588376706951-ap-south-1-an")
AWS_S3_OUTPUT_PREFIX = os.getenv("AWS_S3_OUTPUT_PREFIX","outputs")
AWS_S3_MODEL_BUCKET = os.getenv("AWS_S3_MODEL_BUCKET","models-588376706951-ap-south-1-an")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092")

WEED_REQUEST = "weed.job.created"
WEED_RESULT = "weed.job.result"
DISEASE_REQUEST = "disease.job.created"
DISEASE_RESULT = "disease.job.result"
SOIL_REQUEST = "soil.job.created"
SOIL_RESULT = "soil.job.result"

WEED_MODEL_KEY = "weed/models/weed_model.pth"
PLANT_MODEL_KEY = "disease/models/plant_classifier_model.pth"
DISEASE_MODEL_KEY = "disease/models/disease_model.pth"
PLANT_CLASS_MAPPING_KEY = "disease/labels/plant_class_mapping.json"
DISEASE_CLASS_MAPPING_KEY = "disease/labels/plant_disease_mapping.json"

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR","runtime_models")

DISEASE_IMAGE_SIZE = 224
DISEASE_PLANT_CONFIDENCE_THRESHOLD = 0.80
DISEASE_TOP_K = 5