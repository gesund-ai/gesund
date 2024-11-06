class InputError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)
    
    def __str__(self):
        return self.message


class DataLoadError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)
    
    def __str__(self):
        return self.message
    

class MetricCalculationError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)
    
    def __str__(self):
        return self.message
    

class PlotError(Exception):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(msg)
    
    def __str__(self):
        return self.message
