import json
import logging

from confluent_kafka import Consumer


logger = logging.getLogger(__name__)


class KafkaConsumer:

    def __init__(self, bootstrap_servers, topics):

        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": "inference-group",
            "auto.offset.reset": "earliest"
        })

        self.consumer.subscribe(topics)

        logger.info(
            "Kafka consumer subscribed to topics: %s",
            topics
        )

    def poll(self):

        msg = self.consumer.poll(1.0)

        if msg is None:
            return None

        if msg.error():
            logger.error(
                "Kafka consumer error: %s",
                msg.error()
            )
            return None

        try:
            payload = json.loads(
                msg.value().decode("utf-8")
            )

            logger.info(
                "Received inference job. jobId=%s",
                payload.get("jobId")
            )

            return payload

        except json.JSONDecodeError:
            logger.exception(
                "Failed to decode Kafka message as JSON"
            )
            return None