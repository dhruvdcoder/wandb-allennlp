from typing import TYPE_CHECKING, Optional, Callable
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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
    def training_start(self, trainer: "CallbackTrainer") -> None:
        # This is an ugly hack to get the tensorboard instance to know about the trainer, because
        # the callbacks are defined before the trainer.
        self._get_batch_num_total = lambda: trainer.batch_num_total

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
        """ We would have tried validation after nan. Log current metrics anyway """

        if isinstance(trainer.exception, ValueError):
            if str(trainer.exception).strip() == "nan loss encountered":
                logger.info(
                    "Nan loss encountered. Logging current metrics to wandb")
                self.wandb.log(trainer.metrics)
