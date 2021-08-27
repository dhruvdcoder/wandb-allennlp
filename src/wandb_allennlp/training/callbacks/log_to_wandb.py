from typing import List, Tuple, Union, Dict, Any, Optional, Callable
import logging
from allennlp.common.registrable import Registrable
from allennlp.training.callbacks import (
    WandBCallback,
    TrainerCallback,
)
from allennlp.training import GradientDescentTrainer
from allennlp.data import TensorDict

from allennlp.models.archival import archive_model, verify_include_in_archive
from wandb_allennlp.utils import read_from_env
from overrides import overrides
import os
import torch
from .utils import flatten_dict

logger = logging.getLogger(__name__)


class AllennlpWandbSubCallback(Registrable):
    """
    This is the abstract class that describes a sub-callback to be used with
    AllennlpWandbCallback.

    There will be only one isinstance of AllennlpWandbCallback per trainer.
    To add custom functionallity to this isinstance will require inheritance
    and code duplication. This class is intented to aid extensibility using
    composition.
    """

    def __init__(self, priority: int, **kwargs: Any):
        self.priority = priority

    def on_start_(
        self,
        super_callback: "AllennlpWandbCallback",
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer

    def on_batch_(
        self,
        super_callback: "AllennlpWandbCallback",
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        pass

    def on_epoch_(
        self,
        super_callback: "AllennlpWandbCallback",
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called after the end of each epoch.
        """
        pass

    def on_end_(
        self,
        super_callback: "AllennlpWandbCallback",
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called after the final training epoch.
        """
        pass


@TrainerCallback.register("wandb_allennlp")
class AllennlpWandbCallback(WandBCallback):
    """
    This callback should only be used with `train_with_wandb` command.

    Note:
        If used with `allennlp train` command, this might have unexpected
        behaviour because we read some arguments from environment variables.
    """

    def __init__(
        self,
        serialization_dir: str,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        watch_model: bool = True,
        files_to_save: List[str] = ["config.json", "out.log"],
        files_to_save_at_end: Optional[List[str]] = None,
        include_in_archive: List[str] = None,
        save_model_archive: bool = True,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        sub_callbacks: Optional[List[AllennlpWandbSubCallback]] = None,
    ) -> None:
        logger.debug("Wandb related varaibles")
        logger.debug(
            "%s |   %s  |   %s",
            "variable".ljust(15),
            "value from env".ljust(50),
            "value in constructor".ljust(50),
        )

        for e, a in [("PROJECT", project), ("ENTITY", entity)]:
            logger.debug(
                "%s |   %s  |   %s",
                str(e).lower()[:15].ljust(15),
                str(read_from_env("WANDB_" + e))[:50].ljust(50),
                str(a)[:50].ljust(50),
            )
        logger.debug("All wandb related envirnment varaibles")
        logger.debug("%s |   %s  ", "ENV VAR.".ljust(15), "VALUE".ljust(50))

        for k, v in os.environ.items():
            if "WANDB" in k or "ALLENNLP" in k:
                logger.debug(
                    "%s |   %s  ",
                    str(k)[:15].ljust(15),
                    str(v)[:50].ljust(50),
                )
        t = read_from_env("WANDB_TAGS") or tags

        if isinstance(t, str):
            tags = t.split(",")
        else:
            tags = t
        super().__init__(
            serialization_dir,
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            # prefer env variables because
            project=read_from_env("WANDB_PROJECT") or project,
            entity=read_from_env("WANDB_ENTITY") or entity,
            group=read_from_env("WANDB_GROUP") or group,
            name=read_from_env("WANDB_NAME") or name,
            notes=read_from_env("WANDB_NOTES") or notes,
            tags=tags,
            watch_model=watch_model,
            files_to_save=tuple(files_to_save),
            wandb_kwargs=wandb_kwargs,
        )
        self._files_to_save_at_end = files_to_save_at_end or []
        self.include_in_archive = include_in_archive
        verify_include_in_archive(include_in_archive)
        self.save_model_archive = save_model_archive
        self.priority = 100
        self.sub_callbacks = sorted(
            sub_callbacks or [], key=lambda x: x.priority, reverse=True
        )

        if save_model_archive:
            self._files_to_save_at_end.append("model.tar.gz")
        # do not set wandb dir to be inside the serialization directory.

        if "dir" in self._wandb_kwargs:
            self._wandb_kwargs["dir"] = None

        if "config" in self._wandb_kwargs:
            self._wandb_kwargs["config"] = flatten_dict(
                self._wandb_kwargs["config"]
            )

    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        super().on_start(trainer, is_primary=is_primary, **kwargs)

        for subcallback in self.sub_callbacks:
            subcallback.on_start_(self, trainer, is_primary=is_primary)

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        super().on_batch(
            trainer,
            batch_inputs,
            batch_outputs,
            batch_metrics,
            epoch,
            batch_number,
            is_training,
            is_primary=is_primary,
            batch_grad_norm=batch_grad_norm,
        )

        for sub_callback in self.sub_callbacks:
            sub_callback.on_batch_(
                self,
                trainer,
                batch_inputs,
                batch_outputs,
                batch_metrics,
                epoch,
                batch_number,
                is_training,
                is_primary=is_primary,
                batch_grad_norm=batch_grad_norm,
            )

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        super().on_epoch(
            trainer, metrics, epoch, is_primary=is_primary, **kwargs
        )

        for sub_callback in self.sub_callbacks:
            sub_callback.on_epoch_(
                self, trainer, metrics, epoch, is_primary=is_primary, **kwargs
            )

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        for sub_callback in self.sub_callbacks:
            sub_callback.on_end_(
                self,
                trainer,
                metrics=metrics,
                epoch=epoch,
                is_primary=is_primary,
            )
        super().on_end(
            trainer, metrics=metrics, epoch=epoch, is_primary=is_primary
        )

    @overrides
    def close(self) -> None:
        import wandb

        assert wandb.run is not None
        # set this here for resuming
        os.environ.update({"WANDB_RUN_ID": str(wandb.run.id)})

        if self.save_model_archive:
            # we will have to create archive prematurely here.
            # the `train_model()` in  `allennlp train` will
            # recreate the same model archive later. However,
            # this duplication cannot be avioded at this stage.
            logger.info("Archiving model before closing wandb.")
            archive_model(
                self.serialization_dir,
                include_in_archive=self.include_in_archive,
            )

        if self._files_to_save_at_end:
            for fpath in self._files_to_save_at_end:
                self.wandb.save(  # type: ignore
                    os.path.join(self.serialization_dir, fpath),
                    base_path=self.serialization_dir,
                    policy="end",
                )

        super().close()
