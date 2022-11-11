import logging
import os
import sys

LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
logging.getLogger().setLevel(logging.WARNING)


def create_logger(log_in_console=True, log_in_file=False, log_filename="etl_info.log"):
    """Creates a logger object that logs into console and file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(LOGLEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_in_file:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_in_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = create_logger()


def log(module_split=True, module_msg=""):
    """Decorator that logs function calls, arguments, outputs, and exceptions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if module_split:
                _module_msg = " " + module_msg if module_msg else ""
                logger.info(f"Start{_module_msg}".center(70, "-"))

            # Logging parameters
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            msg = f"Args: {signature}"
            logger.debug(msg)

            try:
                result = func(*args, **kwargs)

                # Logging return
                msg = f"Return: {str(result)}"
                logger.debug(msg)

                if module_split:
                    logger.info(f"Successfully Finished{_module_msg}".center(70, "-"))
                return result
            except Exception as exception:
                logger.exception(f"Exception:".center(70, "!"))
                print("-" * 163)

        return wrapper

    return
