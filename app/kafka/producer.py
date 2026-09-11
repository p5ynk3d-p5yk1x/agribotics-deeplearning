import json
import logging

from confluent_kafka import Producer

logger = logging.getLogger(__name__)


class KafkaProducer:

    def __init__(self, bootstrap_servers):

        logger.info(
            "Initializing Kafka producer with bootstrap servers: %s",
            bootstrap_servers
        )

        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers
        })

        logger.info("Kafka producer initialized successfully")

    def publish(self, topic, payload):

        try:
            logger.info(
                "Starting publish. topic=%s jobId=%s payload=%s",
                topic,
                payload.get("jobId"),
                payload
            )

            message = json.dumps(payload).encode("utf-8")

            logger.info(
                "Payload serialized successfully. size=%d bytes",
                len(message)
            )

            self.producer.produce(
                topic,
                message
            )

            logger.info(
                "Produce request queued. topic=%s jobId=%s",
                topic,
                payload.get("jobId")
            )

            logger.info(
                "Flushing producer. topic=%s jobId=%s",
                topic,
                payload.get("jobId")
            )

            remaining = self.producer.flush()

            logger.info(
                "Flush completed. remaining_messages=%s topic=%s jobId=%s",
                remaining,
                topic,
                payload.get("jobId")
            )

            logger.info(
                "Published event successfully. topic=%s jobId=%s",
                topic,
                payload.get("jobId")
            )

        except Exception as e:
            logger.exception(
                "Failed to publish event. topic=%s jobId=%s error=%s",
                topic,
                payload.get("jobId"),
                str(e)
            )
            raise