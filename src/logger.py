"""For logging execution and errors"""
# Import Python's built-in logging module
import logging
# Import os module for file path operations (like creating log directory)
import os
# Import datetime module for timestamping log files
from datetime import datetime


#Name of the log folder where logs will be stored
LOG_DIR = "logs"
#create log folder if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)



# Build log file name with current date and time like : logs/log_2023-10-05_14-30-00.log(changes Daily )
LOG_FILE = os.path.join(
    LOG_DIR,
    f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

#Configure the ROOT logger once for whole application
logging.basicConfig(
    #Write all logs to this file
    filename=LOG_FILE,
    #Log message format:
    #   - %(asctime)s: Timestamp of the log entry
    #   - %(levelname)s: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    #   - %(message)s: The actual log message
    format="[%(asctime)s] %(levelname)s - %(message)s",
    # Minumum log level to record (INFO)
    level=logging.INFO
)


def get_logger(name) :
    """
    Returns a named logger that inherits the root configuration above.
    Use different names for different modules to differentiate log sources.
    """
    # Get (or create) a logger with the specified name
    logger = logging.getLogger(name)
    #Ensure this logger emits INFO and above messages
    logger.setLevel(logging.INFO)
    # Return the configured logger
    return logger