from typing import List, Tuple, Union, Dict, Any, Optional
import json
import logging
from copy import deepcopy
from pathlib import Path
from allennlp.models.archival import CONFIG_NAME

logger = logging.getLogger(__name__)
Number = Union[int, float]
Value = Union[int, float, bool, str]


def get_allennlp_major_minor_versions() -> Tuple[int, int]:
    import allennlp.version

    return int(allennlp.version._MAJOR), int(allennlp.version._MINOR)


def flatten_dict(params: Dict[str, Any],
                 delimiter: str = ".") -> Dict[str, Value]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a.b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'.'``.
    Returns:
        Flattened dict.
    """
    output: Dict[str, Union[str, Number]] = {}

    def populate(inp: Union[Dict[str, Any], List, str, Number, bool],
                 prefix: List[str]) -> None:

        if isinstance(inp, dict):
            for k, v in inp.items():
                populate(v, deepcopy(prefix) + [k])

        elif isinstance(inp, list):
            for i, val in enumerate(inp):
                populate(val, deepcopy(prefix) + [str(i)])
        elif isinstance(inp, (str, float, int, bool)) or (inp is None):
            output[delimiter.join(prefix)] = inp
        else:  # unsupported type
            raise ValueError(
                f"Unsuported type {type(inp)} at {delimiter.join(prefix)} for flattening."
            )

    populate(params, [])

    return output


def get_config_from_serialization_dir(dir_: str, ) -> Dict[str, Value]:
    with open(Path(dir_) / CONFIG_NAME) as f:
        config_dict = json.load(f)
    config_dict = flatten_dict(config_dict)

    return config_dict
