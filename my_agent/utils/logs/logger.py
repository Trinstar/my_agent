import logging
import os
import time
from typing import Optional


def get_logger(
    name: Optional[str] = None,
    save: bool = False,
    log_file: Optional[str] = None,
    log_file_path: Optional[str] = None,
    log_level: int = logging.WARNING,
    log_format: str = '[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name (Optional[str]): Name of the logger. Defaults to "global_logger".
        save (bool): Whether to save logs to a file. Defaults to False.
        log_file (Optional[str]): Name of the log file. Defaults to None.
        log_file_path (Optional[str]): Path to save the log file. Defaults to None.
        log_level (int): Logging level. Defaults to logging.WARNING.
        log_format (str): Format for log messages. Defaults to a standard format.
        date_format (str): Date format for log messages. Defaults to ISO format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    name = name or "global_logger"
    log_file_name = log_file or f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_file_path or os.path.join(os.getcwd(), "logs")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        if save:
            os.makedirs(log_file_path, exist_ok=True)
            log_file = os.path.join(log_file_path, log_file_name)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

    return logger
