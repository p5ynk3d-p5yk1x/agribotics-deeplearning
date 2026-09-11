import logging
import os
import boto3

logger = logging.getLogger(__name__)

class ModelStorage:
    def __init__(self,bucket: str,region: str,cache_dir: str):
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.client = boto3.client("s3",region_name=region)
        os.makedirs(self.cache_dir,exist_ok=True)
        logger.info("Model storage initialized | Bucket: %s | Cache: %s",bucket,self.cache_dir)

    def get_model(self,key: str) -> str:
        filename = os.path.basename(key)
        local_path = os.path.join(self.cache_dir,filename)

        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            logger.info("Using cached model artifact | Key: %s | Path: %s",key,local_path)
            return local_path

        logger.info("Downloading model artifact from S3 | Bucket: %s | Key: %s",self.bucket,key)

        try:
            self.client.download_file(self.bucket,key,local_path)
        except Exception:
            if os.path.isfile(local_path):
                os.remove(local_path)
            logger.exception("Failed to download model artifact | Key: %s",key)
            raise

        if not os.path.isfile(local_path) or os.path.getsize(local_path) == 0:
            raise RuntimeError(f"Downloaded model artifact is invalid: {key}")

        logger.info("Model artifact downloaded | Key: %s | Path: %s",key,local_path)
        return local_path