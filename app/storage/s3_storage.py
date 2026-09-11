import logging
import boto3
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class S3Storage:
    def __init__(self,bucket: str,region: str,output_prefix: str = "outputs"):
        self.bucket = bucket
        self.region = region
        self.output_prefix = output_prefix.strip("/")
        self.client = boto3.client("s3",region_name=region)
        logger.info("S3 storage initialized | Bucket: %s | Region: %s",bucket,region)

    def download_image(self,key: str) -> np.ndarray:
        logger.info("Downloading image from S3 | Bucket: %s | Key: %s",self.bucket,key)
        response = self.client.get_object(Bucket=self.bucket,Key=key)
        body = response["Body"]
        try:
            image_bytes = body.read()
        finally:
            body.close()

        if not image_bytes:
            raise RuntimeError(f"S3 object is empty: {key}")

        image_array = np.frombuffer(image_bytes,dtype=np.uint8)
        image = cv2.imdecode(image_array,cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Unable to decode S3 image: {key}")

        logger.info("S3 image decoded successfully | Key: %s | Shape: %s",key,image.shape)
        return image

    def upload_image(self,image: np.ndarray,key: str,image_format: str = ".jpg",content_type: str = "image/jpeg") -> str:
        if image is None or not isinstance(image,np.ndarray) or image.size == 0:
            raise ValueError("Output image is empty or invalid")

        success,encoded = cv2.imencode(image_format,image)

        if not success:
            raise RuntimeError(f"Unable to encode output image for S3 key: {key}")

        image_bytes = encoded.tobytes()
        logger.info("Uploading output image to S3 | Bucket: %s | Key: %s | Bytes: %d",self.bucket,key,len(image_bytes))
        self.client.put_object(Bucket=self.bucket,Key=key,Body=image_bytes,ContentType=content_type)
        logger.info("S3 output uploaded successfully | Key: %s",key)
        return key

    def build_output_key(self,job_type: str,job_id: str) -> str:
        return f"{self.output_prefix}/{job_type}/{job_id}"