from typing import List, Tuple, Union, Dict, Any, Optional, Callable
import json
import logging
from .utils import get_config_from_serialization_dir, flatten_dict, get_allennlp_major_minor_versions
from copy import deepcopy
from pathlib import Path
from allennlp.models.archival import CONFIG_NAME

logger = logging.getLogger(__name__)

Number = Union[int, float]
Value = Union[int, float, bool, str]

allennlp_version_major, allennlp_version_minor =\
    get_allennlp_major_minor_versions()

if allennlp_version_major >= 1:
    from allennlp.training.trainer import (  # noqa
        GradientDescentTrainer, BatchCallback, EpochCallback,
    )

    @EpochCallback.register("log_metrics_to_wandb")
    class LogMetricsToWandb(EpochCallback):
        def __init__(
                self,
                epoch_end_log_freq: int = 1,
                watch_model: bool = False,
                watch_log_freq: int = 1000,
                trainer: GradientDescentTrainer = None
        ) -> None:
            # import wandb here to be sure that it was initialized
            # before this line was executed
            super().__init__()
            import wandb  # type: ignore

            self.config: Optional[Dict[str, Value]] = None

            self.wandb = wandb
            self.epoch_end_log_freq = 1
            self.current_batch_num = -1
            self.current_epoch_num = -1
            self.previous_logged_epoch = -1

            self.watch_log_freq = watch_log_freq
            self.watch_model = watch_model
            self.is_watching = False

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

            """

            if is_master and (self.config is None):
                self.update_config(trainer)

            if self.watch_model and not self.is_watching:
                logger.info("Watching trainer model with wandb")
                self.wandb.watch(trainer.model, log_freq=self.watch_log_freq)
                self.is_watching = True

            self.current_epoch_num += 1

            if (is_master
                    and (self.current_epoch_num - self.previous_logged_epoch)
                    >= self.epoch_end_log_freq):
                logger.info("Writing metrics for the epoch to wandb")
                self.wandb.log(
                    {
                        **metrics,
                    },
                    step=self.current_epoch_num,
                )
                self.previous_logged_epoch = self.current_epoch_num

elif allennlp_version_minor >= 8:
    from allennlp.training.callbacks.callback import Callback, handle_event
    from allennlp.training.callbacks.events import Events
    from allennlp.training.callback_trainer import CallbackTrainer

    @Callback.register("log_metrics_to_wandb")
    class LogMetricsToWandb(Callback):
        def __init__(self, debug=False, debug_log_freq: int = 100) -> None:
            # import wandb here to be sure that it was initialized
            # before this line was executed
            super().__init__()
            import wandb  # type: ignore
            self.wandb = wandb
            self.debug = debug
            self.debug_log_freq = debug_log_freq
            self._get_batch_num_total: Optional[Callable[[], int]] = None

        @handle_event(Events.TRAINING_START)
        def training_start(self, trainer: CallbackTrainer) -> None:
            # the callbacks are defined before the trainer.
            self._get_batch_num_total = lambda: trainer.batch_num_total
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            logger.info(f"Sending config to wandb...")
            self.config = get_config_from_serialization_dir(
                trainer._serialization_dir)
            self.wandb.config.update(self.config)

        @handle_event(Events.EPOCH_END, priority=150)  # type: ignore
        def end_of_epoch_log(self, trainer: "CallbackTrainer") -> None:
            """ This should run after all the epoch end metrics have
            been computed by the metric_tracker callback.

            The final callback in metric_tracker for EPOCH_END event
            has priority 100. So this callback should have priority > 100
            to be executed after
            """
            logger.info("Writing metrics to wandb")
            self.wandb.log(trainer.metrics)

        @handle_event(Events.BATCH_END, priority=1000)  # type: ignore
        def end_of_batch_log(self, trainer: "CallbackTrainer") -> None:
            """Log metrics after each mini batch if in debug mode

            The final callback in metric_tracker for EPOCH_END event
            has priority 100. So this callback should have priority > 100
            to be executed after
            """

            if self.debug:
                if self._get_batch_num_total() % self.debug_log_freq == 0:
                    logger.info("Writing metrics to wandb")
                    self.wandb.log(trainer.metrics)

        @handle_event(Events.ERROR, priority=150)
        def log_after_crash(self, trainer: "CallbackTrainer") -> None:
            """ We would have tried validation after nan. Log current metrics
                anyway
            """

            if isinstance(trainer.exception, ValueError):
                if str(trainer.exception).strip() == "nan loss encountered":
                    logger.info(
                        "Nan loss encountered. Logging current metrics to wandb"
                    )
                    self.wandb.log(trainer.metrics)

else:
    raise RuntimeError(
        f"AllenNLP version {allennlp_version_major}.{allennlp_version_minor} not supported by wandb-allennlp"
    )
