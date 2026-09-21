from functools import cache
from importlib.metadata import version

DISTRIBUTION_NAME = "fraud-detector"


@cache
def get_app_version() -> str:
    return version(DISTRIBUTION_NAME)
