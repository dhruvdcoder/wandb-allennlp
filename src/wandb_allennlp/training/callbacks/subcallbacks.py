from typing import List, Tuple, Union, Dict, Any, Optional

from wandb_allennlp.training.callbacks.log_to_wandb import (
    AllennlpWandbSubCallback,
    AllennlpWandbCallback,
    GradientDescentTrainer,
)


@AllennlpWandbSubCallback.register("log_best_validation_metrics")
class LogBestValidationMetrics(AllennlpWandbSubCallback):
    def on_epoch_(
        self,
        super_callback: AllennlpWandbCallback,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Logs the best_validation_* to wandb.
        """
        # identify the tracked metric
        # this logic fragile and depends on an internal varaible of the trainer.
        # Hence it will need to be updated with newer versions of allennlp.
        metric_names_to_take = [
            f"best_validation_{name}"
            for sign, name in trainer._metric_tracker.tracked_metrics
        ]

        super_callback.log_scalars(
            {
                name.replace("validation_", "", 1): value
                for name, value in metrics.items()
                for metric_name in metric_names_to_take
                if name == metric_name
            },
            log_prefix="validation",
            epoch=epoch,
        )
