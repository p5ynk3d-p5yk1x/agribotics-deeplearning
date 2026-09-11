import logging
from app.config import AWS_REGION,AWS_S3_MODEL_BUCKET,AWS_S3_OUTPUT_PREFIX,AWS_S3_UPLOAD_BUCKET,DISEASE_CLASS_MAPPING_KEY,DISEASE_IMAGE_SIZE,DISEASE_MODEL_KEY,DISEASE_PLANT_CONFIDENCE_THRESHOLD,DISEASE_REQUEST,DISEASE_RESULT,DISEASE_TOP_K,KAFKA_BOOTSTRAP_SERVERS,MODEL_CACHE_DIR,PLANT_CLASS_MAPPING_KEY,PLANT_MODEL_KEY,SOIL_REQUEST,SOIL_RESULT,WEED_MODEL_KEY,WEED_REQUEST,WEED_RESULT
from app.kafka.consumer import KafkaConsumer
from app.kafka.producer import KafkaProducer
from app.model.disease_detection.diseaseInference import DiseaseInferenceEngine
from app.model.soil.soil_detection import analyze_soil_event
from app.model.weed_detection.inference import InferenceEngine
from app.storage.model_storage import ModelStorage
from app.storage.s3_storage import S3Storage

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Initializing Agribotics inference service")

model_storage = ModelStorage(AWS_S3_MODEL_BUCKET,AWS_REGION,MODEL_CACHE_DIR)
logger.info("Resolving inference model artifacts")

weed_model_path = model_storage.get_model(WEED_MODEL_KEY)
plant_model_path = model_storage.get_model(PLANT_MODEL_KEY)
disease_model_path = model_storage.get_model(DISEASE_MODEL_KEY)
plant_mapping_path = model_storage.get_model(PLANT_CLASS_MAPPING_KEY)
disease_mapping_path = model_storage.get_model(DISEASE_CLASS_MAPPING_KEY)

logger.info("All inference model artifacts resolved successfully")

weed_engine = InferenceEngine(weed_model_path)
disease_engine = DiseaseInferenceEngine(plant_model_path=plant_model_path,disease_model_path=disease_model_path,plant_mapping_path=plant_mapping_path,disease_mapping_path=disease_mapping_path,image_size=DISEASE_IMAGE_SIZE,plant_confidence_threshold=DISEASE_PLANT_CONFIDENCE_THRESHOLD,top_k=DISEASE_TOP_K)

logger.info("All inference models loaded successfully")

storage = S3Storage(AWS_S3_UPLOAD_BUCKET,AWS_REGION,AWS_S3_OUTPUT_PREFIX)
logger.info("Storage service initialized")

consumer = KafkaConsumer(KAFKA_BOOTSTRAP_SERVERS,[WEED_REQUEST,DISEASE_REQUEST,SOIL_REQUEST])
producer = KafkaProducer(KAFKA_BOOTSTRAP_SERVERS)

logger.info("Waiting for Kafka inference events")

while True:
    event = consumer.poll()

    if event is None:
        continue

    job_id = event.get("jobId")
    job_type = event.get("jobType")

    try:
        if not job_id:
            raise ValueError("Kafka event does not contain jobId")

        if not job_type:
            raise ValueError("Kafka event does not contain jobType")

        if job_type not in ("WEED","DISEASE","SOIL"):
            raise ValueError(f"Unsupported job type: {job_type}")

        logger.info("Processing inference job | Job ID: %s | Job type: %s",job_id,job_type)

        if job_type == "WEED":
            image_key = event.get("imagePath")

            if not image_key:
                raise ValueError("WEED event does not contain imagePath")

            logger.info("Downloading WEED input image | Job ID: %s | Input key: %s",job_id,image_key)

            image = storage.download_image(image_key)
            result = weed_engine.predict(image)

            producer.publish(WEED_RESULT,{
                "jobId": job_id,
                "status": "COMPLETED",
                "jobType": job_type,
                "resultPayload": {
                    "prediction": result["weedName"],
                    "confidence": result["confidence"]
                }
            })

        elif job_type == "DISEASE":
            image_key = event.get("imagePath")

            if not image_key:
                raise ValueError("DISEASE event does not contain imagePath")

            logger.info("Downloading DISEASE input image | Job ID: %s | Input key: %s",job_id,image_key)

            image = storage.download_image(image_key)
            result,annotated_image = disease_engine.predict(image=image,job_id=job_id)

            output_key = storage.build_output_key(job_type,job_id)

            storage.upload_image(image=annotated_image,key=output_key,image_format=".jpg",content_type="image/jpeg")

            result["annotatedImagePath"] = output_key

            producer.publish(DISEASE_RESULT,{
                "jobId": job_id,
                "status": "COMPLETED",
                "jobType": job_type,
                "resultPayload": result
            })

        elif job_type == "SOIL":
            result = analyze_soil_event(event)

            producer.publish(SOIL_RESULT,{
                "jobId": job_id,
                "status": "COMPLETED",
                "jobType": job_type,
                "resultPayload": result
            })

        logger.info("Inference job completed | Job ID: %s | Job type: %s",job_id,job_type)

    except Exception as error:
        logger.exception("Inference job failed | Job ID: %s | Job type: %s",job_id,job_type)

        if job_type == "WEED":
            result_topic = WEED_RESULT
        elif job_type == "DISEASE":
            result_topic = DISEASE_RESULT
        elif job_type == "SOIL":
            result_topic = SOIL_RESULT
        else:
            logger.error("Cannot publish failure because job type is unsupported | Job ID: %s | Job type: %s",job_id,job_type)
            continue

        producer.publish(result_topic,{
            "jobId": job_id,
            "jobType": job_type,
            "status": "FAILED",
            "failureReason": str(error)
        })