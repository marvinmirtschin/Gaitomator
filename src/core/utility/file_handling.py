import glob
import json
import os
import warnings
from pathlib import Path

MATCH_ANY = "*"
MATCH_ANY_INCLUDE_SUB_DIR = "**"
# use os.path.join to keep it os independent
KNOWN_PATHS = [os.path.join("Documents", "Projects", "training-data"),
               os.path.join("Documents", "Projects", "neXenio", "Android", "training-data"),
               os.path.join("Developer", "training-data"),
               os.path.join("Documents", "01_projects", "training-data"),
               os.path.join("usr", "src", "app", "training-data")]  # soc-lab folder

USE_ALL_FILES_FROM_DEVICE_ID = 0


def get_current_working_directory():
    return os.getcwd()


def get_current_directory():
    return os.path.dirname(os.path.abspath("__file__"))


def get_parent_directory():
    return get_parent_directory_for(get_current_directory())


def get_parent_directory_for(directory):
    return os.path.abspath(os.path.join(directory, os.pardir))


def get_home_path():
    return os.path.expanduser("~")


def get_project_directory():
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_directory():
    return os.path.join(get_project_directory(), "Data")


def get_generated_data_directory():
    return os.path.join(get_data_directory(), "generated")


def get_python_directory():
    return os.path.join(get_project_directory(), "Python")


def get_report_directory():
    return os.path.join(get_project_directory(), "reports")


def get_tests_path():
    return os.path.join(get_python_directory(), "tests")


def get_all_json_files_for_directory(directory, print_file_names=False):
    return get_file_names_in_directory_for_pattern(directory, "*.json", print_file_names)


def get_file_names_in_resource_directory_for_pattern(requested_file_pattern, print_file_names=False):
    return get_file_names_in_directory_for_pattern(get_data_directory(), requested_file_pattern, print_file_names)


def get_file_names_in_directory_for_pattern(directory, pattern, print_file_names=False):
    try:
        requested_file_pattern = os.path.join(directory, MATCH_ANY_INCLUDE_SUB_DIR, pattern)
    except TypeError:
        raise Exception("NoneType Error for directory {dir} or pattern {pattern}"
                        .format(dir=directory, pattern=pattern))

    filtered_file_names = glob.glob(requested_file_pattern, recursive=True)
    if len(filtered_file_names) == 0:
        warnings.warn("No files matched the given pattern: {}".format(requested_file_pattern), stacklevel=2)

    if print_file_names:
        for file_name in filtered_file_names:
            short_file_name = file_name.split(os.path.sep)[-1]
            print(short_file_name)
    return filtered_file_names


def get_or_create_html_visualizations_directory():
    path = get_html_visualizations_path()
    os.makedirs(path, exist_ok=True)
    return path


def get_html_visualizations_path():
    return os.path.join(get_report_directory(), "html")


"""parse file description from file name"""


def get_json_string_from_file(file_name):
    try:
        with open(file_name, "r") as file_:
            data_text = file_.read()
        try:
            json_string = json.loads(data_text)
            return json_string
        except json.JSONDecodeError as e:
            raise Exception("Decoding json failed for {file_name}: {error}".format(file_name=file_name, error=e))
    except FileNotFoundError as e:
        raise Exception("File {file_name} not found: {error}".format(file_name=file_name, error=e))
