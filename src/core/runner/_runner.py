import hashlib
import json
import os
from typing import Any, Dict

import pandas as pd

from src.core.utility import file_handling


class ImplementationRunner:
    """
    Use this to configure your transformation pipeline, run the transformation and print the classification results.
    """

    def __init__(self, data_accessor, use_available_results=True):
        self.data_accessor = data_accessor
        self.use_available_results = use_available_results

    def process_data(self, pipeline=None, **kwargs):
        """
        Transform the data return by the DataAccessor with the given pipeline steps. If self.use_available_results is True, this will look for
        results of a matching configuration and return it. If no results are available, normal process will be performed and the result will be
        saved to be used in the future.

        Parameters
        ----------
        pipeline: sklearn.pipeline.Pipeline, default=None
            The pipeline used to transform the data returned by the DataAccessor. If non is given, the default pipeline returned by
            self.get_pipeline() will be used.

        Returns
        -------
        pd.DataFrame
            Transformed data frame
        """
        if pipeline is None:
            pipeline = self.get_pipeline(recording_frequency=self.data_accessor.get_frequency(), **kwargs)
        configuration = self.get_configuration(pipeline)
        hash_id = dict_hash(configuration)

        directory = os.path.join(file_handling.get_data_directory(), "self_created", "progress", self.get_paper_name(), hash_id)
        data_file = os.path.join(directory, "values.csv")
        if self.use_available_results and os.path.exists(directory):
            transformed_data_frame = self.read_data_frame(data_file)
        else:
            transformed_data_frame: pd.DataFrame = pipeline.transform(self.data_accessor.transform())

            if self.use_available_results:
                os.makedirs(directory, exist_ok=True)
                self.save_data_frame(transformed_data_frame, data_file)

                # save configuration
                configuration_file = os.path.join(directory, "configuration.txt")
                with open(configuration_file, 'w') as f:
                    json.dump(configuration, f)

        return transformed_data_frame

    def read_data_frame(self, data_file):
        """
        Read a saved transformed data frame from a file.

        Parameters
        ----------
        data_file: str
            Path to file
        """
        return pd.read_csv(data_file, index_col=0)

    def save_data_frame(self, transformed_data_frame, data_file):
        """
        Save the transformed data frame into a file.

        Parameters
        ----------
        transformed_data_frame: pd.DataFrame
            DataFrame which was returned by self.process_data
        data_file: str
            Path to file
        """
        transformed_data_frame.to_csv(data_file)

    def print_classification_report(self, transformed_data_frame):
        """
        Print classification result. This should use a ClassificationRunner.
        Parameters
        ----------
        transformed_data_frame: pd.DataFrame
            DataFrame which was returned by self.process_data
        """
        raise NotImplementedError("Implement the 'print_classification_report(self, transformed_data_frame)' method in your class.")

    @staticmethod
    def get_paper_name():
        """
        Return name of your implementation, e.g. the name of the paper or author.
        Returns
        -------
            Readable identifier for this runner.
        """
        raise NotImplementedError("Implement the 'get_paper_name()' method in your class.")

    def get_configuration(self, pipeline):
        """
        Return relevant parameters used for getting and transforming data. Override this method to remove irrelevant properties to keep
        configuration clean.

        Parameters
        ----------
        pipeline: sklearn.pipeline.Pipeline
            The pipeline used to transform the data returned by the DataAccessor.

        Returns
        -------
            Dictionary with relevant configuration settings.
        """
        params = pipeline.get_params(deep=True)
        # These contain real transformer classes which are difficult to serialize and not needed here
        steps = params["steps"]
        step_names = [name for name, value in steps]
        keys_to_remove = [key for key in params.keys() if key in step_names]
        keys_to_remove += [key for key in params.keys() if key.endswith("transformer")]
        keys_to_remove += [key for key in params.keys() if key.endswith("error_handler")]

        for key in keys_to_remove:
            try:
                del params[key]
            except KeyError:
                pass

        # internal pipeline parameter
        del params["steps"]
        del params["memory"]
        del params["verbose"]

        return {**params, **self.data_accessor.get_params(deep=True)}

    def get_pipeline(self, recording_frequency, **kwargs):
        """
        Get the default pipeline to be used for this runner.

        Parameters
        ----------
        recording_frequency: int
            Frequency of data returned by self.data_accessor

        Returns
        -------
        sklearn.pipeline.Pipeline
            The pipeline used to transform the data returned by the DataAccessor.
        """
        raise NotImplementedError("Implement the 'get_pipeline(self, **kwargs)' method in your class.")

    def run(self):
        self.print_classification_report(self.process_data(data_accessor=self.data_accessor))


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
