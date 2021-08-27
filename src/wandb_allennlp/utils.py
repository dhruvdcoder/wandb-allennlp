from typing import List, Tuple, Union, Dict, Any, Optional, Callable
import os


def read_from_env(
    key: str, type_: Optional[Callable[[str], Any]] = None
) -> Optional[Any]:
    val_str = os.environ.get(key, None)

    if type_ is not None and isinstance(val_str, str):
        val = type_(val_str)
    else:
        val = val_str

    return val
