import sys
from utils.exception import CustomException

try:
    a = 10 / 0
except Exception as e:
    raise CustomException("Division test failed", sys)
