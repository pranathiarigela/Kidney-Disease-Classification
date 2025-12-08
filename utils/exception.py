import sys
from utils.logger import log_error

class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        super().__init__(message)
        _, _, exc_tb = error_detail.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
        self.message = message

    def __str__(self):
        return f"{self.message} (Error in {self.filename} at line {self.lineno})"
