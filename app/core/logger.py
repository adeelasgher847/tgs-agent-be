import logging
import sys
from app.core.config import settings

def setup_logging():
    """
    Configure the logging system for the application.
    Sets up a console handler with a specific format.
    """
    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create a root logger
    logger = logging.getLogger()
    
    # Set the log level based on settings or default to INFO
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(log_level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
    
    # Set logging level for some noisy 3rd party libraries to WARNING
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    return logger

# Create a module-level logger instance for easy import
logger = logging.getLogger("tgs_agent")
