import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_PARAMETERS = (
    "nitrogenLevel",
    "potassiumLevel",
    "phosphorousLevel",
    "organicCarbonLevel",
)

OPTIONAL_PARAMETERS = (
    "ironLevel",
    "zincLevel",
    "manganeseLevel",
    "copperLevel",
    "boronLevel",
    "sulphurLevel",
    "salinityLevel",
    "electricalConductivity",
    "pH",
)

ALL_PARAMETERS = REQUIRED_PARAMETERS + OPTIONAL_PARAMETERS

def analyze_soil_event(event: dict[str,Any]) -> dict[str,str]:
    logger.info(
        "Processing soil analysis | Job ID: %s | Latitude: %s | Longitude: %s",
        event.get("jobId"),
        event.get("latitude"),
        event.get("longitude"),
    )

    _validate_event(event)

    job_id = event["jobId"]
    rng = random.Random(job_id)
    result_payload: dict[str,str] = {}

    for parameter in ALL_PARAMETERS:
        value = event.get(parameter)

        if value is None:
            continue

        numeric_value = float(value)
        minimum_threshold = max(numeric_value - 25,0)
        maximum_threshold = numeric_value + 25
        optimum_threshold = rng.uniform(minimum_threshold,maximum_threshold)

        difference = numeric_value - optimum_threshold

        if difference >= 15:
            level = "high"
        elif difference < -15:
            level = "depleted"
        else:
            level = "moderate"

        result_payload[parameter] = level

        logger.info(
            "Soil parameter analyzed | Job ID: %s | Parameter: %s | Value: %.4f | Optimum: %.4f | Difference: %.4f | Level: %s",
            job_id,
            parameter,
            numeric_value,
            optimum_threshold,
            difference,
            level,
        )

    logger.info("Soil analysis completed | Job ID: %s | Result: %s",job_id,result_payload)
    return result_payload

def _validate_event(event: dict[str,Any]) -> None:
    if not event.get("jobId"):
        raise ValueError("Soil event does not contain jobId")

    if event.get("jobType") != "SOIL":
        raise ValueError(f"Invalid soil job type: {event.get('jobType')}")

    if event.get("latitude") is None:
        raise ValueError("Soil event does not contain latitude")

    if event.get("longitude") is None:
        raise ValueError("Soil event does not contain longitude")

    for parameter in REQUIRED_PARAMETERS:
        if event.get(parameter) is None:
            raise ValueError(f"Soil event does not contain required parameter: {parameter}")

    for parameter in ALL_PARAMETERS:
        value = event.get(parameter)

        if value is None:
            continue

        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError(f"Soil parameter {parameter} must be numeric")

        if value < 0:
            raise ValueError(f"Soil parameter {parameter} cannot be negative")