import configparser
import logging.config


def configure_logging(log_file_path: str):
    config = configparser.ConfigParser()
    config.read("logger.conf")

    config.set("handler_fileHandler", "args", f"('{log_file_path}', 'a')")
    print(config["handler_fileHandler"]["args"])

    logging.config.fileConfig(config, disable_existing_loggers=False)
