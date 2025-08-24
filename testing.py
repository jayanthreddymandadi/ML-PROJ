from src.logger import get_logger
from src.exception import exception as CustomException
import sys
logger = get_logger(__name__)
def divide_numbers(a, b):
    try:
        result = a / b
        logger.info(f"Division successful: {a} / {b} = {result}")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise CustomException("An unexpected error occurred during division.", sys) from e
if __name__ == "_main_":
    try:
        logger.info("Starting division tests.")
        divide_numbers(10,0)  # Should log error and raise CustomException
    except CustomException as ce:
        logger.error(f"CustomException caught in main: {ce}")