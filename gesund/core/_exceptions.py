class CustomError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)

    def __str__(self):
        return self.message


class InputError(CustomError):
    pass


class DataLoadError(CustomError):
    pass


class MetricCalculationError(CustomError):
    pass


class PlotError(CustomError):
    pass


class FunctionNotFoundError(CustomError):
    pass


class RegistrationNotAllowed(CustomError):
    pass
