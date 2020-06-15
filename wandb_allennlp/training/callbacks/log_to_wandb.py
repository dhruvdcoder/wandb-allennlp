from typing import List, Tuple, Union, Dict, Any, Optional
import logging
from allennlp.training.trainer import (
    GradientDescentTrainer,
    BatchCallback,
    EpochCallback,
)

logger = logging.getLogger(__name__)

@EpochCallback.register('wandb')
class LogMetricsToWandb(EpochCallback):
    def __init__(
        self, epoch_end_log_freq: int = 1, batch_end_log_freq: int = 100
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        import wandb  # type: ignore

        self.wandb = wandb
        self.epoch_end_log_freq = epoch_end_log_freq
        self.batch_end_log_freq = batch_end_log_freq
        self.current_batch_num = -1
        self.current_epoch_num = -1
        self.previous_logged_epoch = -1

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

        self.current_epoch_num += 1

        if (
            is_master
            and (self.current_epoch_num - self.previous_logged_epoch)
            >= self.epoch_end_log_freq
        ):
            logger.info("Writing metrics for the epoch to wandb")
            self.wandb.log(
                {
                    **metrics,
                },
                step=self.global_step,
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
