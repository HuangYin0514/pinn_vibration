import logging
import os


class Logger(logging.Logger):
    '''Inherit from logging.Logger.
    Print logs to console and file.
    Add functions to draw the training log curve.'''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__('Training logger')

        # Print logs to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set the console handler to display all levels

        # Print logs to file
        file_handler_info = logging.FileHandler(os.path.join(self.dir_path, 'train_log.log'))
        file_handler_info.setLevel(logging.INFO)  # Only INFO level and above will be written to this file

        file_handler_error = logging.FileHandler(os.path.join(self.dir_path, 'infer_log.log'))
        file_handler_error.setLevel(logging.DEBUG)  # Only ERROR level and above will be written to this file

        log_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

        console_handler.setFormatter(log_format)
        file_handler_info.setFormatter(log_format)
        file_handler_error.setFormatter(log_format)

        self.addHandler(console_handler)
        self.addHandler(file_handler_info)
        self.addHandler(file_handler_error)


# Example usage
if __name__ == "__main__":
    logger = Logger("logs")
    logger.debug("This is a debug message")  # This will not be written to any file
    logger.info("This is an info message")    # This will be written to info.log
    logger.warning("This is a warning message")  # This will not be written to any file
    logger.error("This is an error message")  # This will be written to error.log
