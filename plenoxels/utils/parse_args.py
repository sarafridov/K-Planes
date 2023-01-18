from typing import Optional


def parse_optfloat(val, default_val=None) -> Optional[float]:
    if val == "None" or val is None:
        return default_val
    return float(val)


def parse_optint(val, default_val=None) -> Optional[int]:
    if val == "None" or val is None:
        return default_val
    return int(val)
