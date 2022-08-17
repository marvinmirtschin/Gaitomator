class ClassificationRunner:
    """
    Use this class to setup your classification scenario.
    """

    def get_classifiers(self):
        """
        Returns the classifiers used in this scenario.

        Returns
        -------
        list
            list of classifiers to perform classification with
        """
        raise NotImplementedError("Implement the 'get_classifiers()' method in your class.")

    def print_classifications(self, data):
        """
        Perform the step to prepare your test and train data and run it with each classifier. After that print the results.
        Parameters
        ----------
        data: pd.DataFrame
            Transformed data frame to be used in the classification
        """
        raise NotImplementedError("Implement the 'print_classifications(data)' method in your class.")
