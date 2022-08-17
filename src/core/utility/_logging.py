import logging
import os
import sys
import warnings
from datetime import datetime

import src.core.utility.file_handling
from src.core.constants import LABEL_RECORD_KEY, LABEL_USER_KEY
from src.core.utility import time_unit
from src.core.utility.data_frame_transformer import DataFrameFeatureUnion


# Note: For yet unknown reasons logger.warning is needed to actually print messages
def get_logger(directory, log_to_file=True):
    directory = os.path.join(directory, "logs")
    os.makedirs(directory, exist_ok=True)
    timestamp = time_unit.get_current_time()
    file_name = os.path.join(directory, str(datetime.fromtimestamp(timestamp)) + ".log")

    logger = logging.getLogger('paper_runner')
    logger.setLevel(logging.NOTSET)
    if log_to_file:
        fh = logging.FileHandler(file_name)
        fh.setLevel(logging.NOTSET)
        logger.addHandler(fh)
    return logger


def log_pipeline(logger, pipeline):
    for key, transformer in pipeline.steps:
        if isinstance(transformer, DataFrameFeatureUnion):
            logger.warning("Feature Pipeline Start")
            for k, v in transformer.transformer_list:
                logger.warning("Transformer {}: {}".format(k, v.__dict__))
            logger.warning("Feature Pipeline End")
        logger.warning("Transformer {}: {}".format(key, transformer.__dict__))


def log_file_infos(logger, transformed_data_frame):
    users = set(transformed_data_frame[LABEL_USER_KEY])
    user_record_set = set(transformed_data_frame[[LABEL_USER_KEY, LABEL_RECORD_KEY]])

    logger.warning("Users: {}".format(users))
    logger.warning("Record-User Pair: {}".format(user_record_set))


def get_system_information():
    import platform

    print(f"OS: {platform.system()}")  # e.g. Windows, Linux, Darwin
    print(f"Bit-System and Linkage: {platform.architecture()}")  # e.g. 64-bit
    print(f"Machine: {platform.machine()}")  # e.g. x86_64
    # print("Hostname: " + platform.node())  # Hostname
    print(f"Processor: {platform.processor()}")  # e.g. i386


def get_software_information():
    """
    Print software information regarding python and the 'requirements.txt'.
    """
    print("Current version of Python is ", sys.version)
    file_path = os.path.join(src.core.utility.file_handling.get_python_directory(), "src", "requirement.txt")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            separator = None
            if "==" in line:
                separator = "=="
            elif ">=" in line:
                separator = ">="

            if separator:
                module_name, requested_module_version = line.split(separator)
            else:
                module_name = line

            module_name = map_module_name(module_name)
            try:
                __import__(module_name)  # ModuleNotFoundError
                # noinspection PyUnresolvedReferences
                version = sys.modules.get(module_name).__version__
                # version = module_name.__version__
                print(module_name + "==" + version)
            except ModuleNotFoundError:
                print("Unable to import: " + module_name)
                warnings.warn("You should correctly install " + module_name + " or add an importable module name for it in "
                                                                              "src.utility._logging#map_module_name")
            except AttributeError:
                print("Unable to verify version for: " + module_name)


def map_module_name(module_name):
    """
    Map the name of python modules given in the 'requirements.txt' to their importable names.

    Note: If you add a module you might need to add it here to see it in your software requirements.

    Parameters
    ----------
    module_name: str
        Name of the module.

    Returns
    -------
        Term for the module which can be used in import.
    """
    if module_name == "scikit-learn":
        return "sklearn"
    if module_name == "PyWavelets":
        return "pywt"
    return module_name


def test_information():
    get_system_information()
    get_software_information()
