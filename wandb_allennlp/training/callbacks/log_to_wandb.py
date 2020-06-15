from typing import List, Tuple, Union, Dict, Any, Optional
import json
import logging
from copy import deepcopy
from pathlib import Path
from allennlp.models.archival import CONFIG_NAME
from allennlp.training.trainer import (
    GradientDescentTrainer,
    BatchCallback,
    EpochCallback,
)

logger = logging.getLogger(__name__)
Number = Union[int, float]
Value = Union[int, float, bool, str]


def _flatten_dict(params: Dict[str, Any], delimiter: str = ".") -> Dict[str, Value]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a.b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    """
    output: Dict[str, Union[str, Number]] = {}

    def populate(
        inp: Union[Dict[str, Any], List, str, Number, bool], prefix: List[str]
    ) -> None:
        if isinstance(inp, dict):
            for k, v in inp.items():
                populate(v, deepcopy(prefix) + [k])

        elif isinstance(inp, list):
            for i, val in enumerate(inp):
                populate(val, deepcopy(prefix) + [str(i)])
        elif isinstance(inp, (str, float, int, bool)):
            output[delimiter.join(prefix)] = inp
        else:  # unsupported type
            raise ValueError(
                f"Unsuported type {type(inp)} at {delimiter.join(prefix)} for flattening."
            )

    populate(params, [])

    return output


def get_config_from_serialization_dir(dir_: str,) -> Dict[str, Value]:
    with open(Path(dir_) / CONFIG_NAME) as f:
        config_dict = json.load(f)
    config_dict = _flatten_dict(config_dict)

    return config_dict


@EpochCallback.register("wandb")
class LogMetricsToWandb(EpochCallback):
    def __init__(
        self, epoch_end_log_freq: int = 1, batch_end_log_freq: int = 100
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        import wandb  # type: ignore

        self.config: Optional[Dict[str, Value]] = None

        self.wandb = wandb
        self.epoch_end_log_freq = epoch_end_log_freq
        self.batch_end_log_freq = batch_end_log_freq
        self.current_batch_num = -1
        self.current_epoch_num = -1
        self.previous_logged_epoch = -1

    def update_config(self, trainer: GradientDescentTrainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            logger.info(f"Sending config to wandb...")
            self.config = get_config_from_serialization_dir(
                trainer._serialization_dir)
            self.wandb.config.update(self.config)

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        """ This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.

        The final callback in metric_tracker for EPOCH_END event
        has priority 100. So this callback should have priority > 100
        to be executed after
        """

        if self.config is None:
            self.update_config(trainer)

        self.current_epoch_num += 1

        if (
            is_master
            and (self.current_epoch_num - self.previous_logged_epoch)
            >= self.epoch_end_log_freq
        ):
            logger.info("Writing metrics for the epoch to wandb")
            self.wandb.log(
                {**metrics, }, step=self.current_epoch_num,
            )
            self.previous_logged_epoch = self.current_epoch_num


#    def end_of_batch_log(
#        self,
#        trainer: GradientDescentTrainer,
#        batch_inputs: List[List[TensorDict]],
#        batch_outputs: List[Dict[str, Any]],
#        epoch: int,
#        batch_number: int,
#        is_training: bool,
#        is_master: bool,
#    ) -> None:
#        """Log metrics after each mini batch if in debug mode
#
#        The final callback in metric_tracker for EPOCH_END event
#        has priority 100. So this callback should have priority > 100
#        to be executed after
#        """
#        pass
#        # self.current_batch_num += 1
#        # self.global_step += 1
#
#        # if (
#        #    is_master
#        #    and (self.global_step - self.previous_logged_global_step)
#        #    >= self.batch_end_log_freq
#        # ):
#        #    logger.info("Writing metrics for the batch to wandb")
#        #    self.wandb.log(
#        #        {
#        #            **metrics,
#        #            "epoch_num": self.current_epoch_num,
#        #            "batch_num": self.current_batch_num,
#        #        },
#        #        step=self.global_step,
#        #    )
#        #    self.previous_logged_epoch = self.current_epoch_num
#
#    def log_after_crash(self, trainer: GradientDescentTrainer) -> None:
#        """ We would have tried validation after nan. Log current metrics anyway """
#        pass
#
