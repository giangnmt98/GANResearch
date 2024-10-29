"""
A module for custom logging with Python's logging library.

This module sets up a custom logger that can log messages to both the console
and a rotating file handler if a log file path is provided. The logger is
highly configurable and suitable for production-level projects.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog


class CustomLogger:
    """
    A custom logger class to log messages to the console and/or a log file.

    Attributes:
        logger (logging.Logger): The main logger instance used to log messages.
    """

    def __init__(self, name, log_file=False, level=logging.INFO, debug=False):
        """
        Initialize the CustomLogger instance.

        Args:
            name (str): The name of the logger.
            log_file (str, optional): The path to the log file. If provided,
                logs will be written to this file. Defaults to None.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.debug = debug

        # Ensure no repeated handlers
        if not self.logger.hasHandlers():
            # Log formatting with detailed information and colors
            log_colors = {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
            if self.debug:
                formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    " - [in %(filename)s:%(lineno)d | %(funcName)s()]",
                    log_colors=log_colors,
                )
            else:
                formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    log_colors=log_colors,
                )
            # Handler to log to console
            # Handler to log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Handler to log to file, if log_file is provided
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a", encoding="utf-8"):
                    pass
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=5 * 1024 * 1024, backupCount=3
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Get the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def log(self, level, message):
        """
        Log a message at the specified logging level.

        Args:
            level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
            message (str): The message to be logged.
        """
        self.logger.log(level, message)
