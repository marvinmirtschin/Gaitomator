# Helper Classes
import warnings


class PipelineErrorHandler:
    instance = None

    def __init__(self):
        self.steps = list()

    def new_case(self, caller=None, allowed_exceptions=None, print_traceback=False):
        current_step = "unknown"
        if len(self.steps) != 0:
            if caller is not self.steps[-1]:
                self.steps.append(caller)

            current_step = str(self.steps[-1])

        return _Case(allowed_exceptions, current_step, print_traceback)

    @staticmethod
    def get_instance():
        if PipelineErrorHandler.instance is None:
            PipelineErrorHandler.instance = PipelineErrorHandler()
        return PipelineErrorHandler.instance


class _Case:

    def __init__(self, allowed_exception, current_step, print_traceback):
        self.allowed_exceptions = allowed_exception
        self.current_step = current_step
        self.print_traceback = print_traceback

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            return True

        if self.allowed_exceptions is not None and exc_type in self.allowed_exceptions:
            warnings.warn("There was an exception in {}: {}".format(self.current_step, exc_value))
            if self.print_traceback:
                warnings.warn(exc_traceback)
            return True
