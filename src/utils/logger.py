import logging
import sys

class Logger:
    def __init__(self, log_dir=None):
        self.logger = logging.getLogger(__name__)
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.INFO)
        
        # Handler for regular info logs -> stdout/output.log
        info_handler = logging.StreamHandler(sys.stdout)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        
        # Handler for warnings and errors -> stderr/error.log
        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        self.logger.addHandler(info_handler)
        self.logger.addHandler(error_handler)

    # Keep old log method for backward compatibility
    def log(self, message):
        self.info(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
