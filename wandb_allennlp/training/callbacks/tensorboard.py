from tensorboardX import SummaryWriter
import inspect
from typing import List, Tuple, Union, Dict, Any, Optional
import json
import logging
from copy import deepcopy
import torch
from pathlib import Path
from allennlp.models.archival import CONFIG_NAME
from allennlp.data.dataloader import TensorDict
from allennlp.training.trainer import (
    GradientDescentTrainer,
    BatchCallback,
    EpochCallback,
)

logger = logging.getLogger(__name__)


def _get_model_args(model: torch.nn.Module) -> List[str]:
    """ Get the mandatory args to model's forward method"""
    with_self = inspect.getfullargspec(model.forward)[0]
    assert with_self[0] == "self"

    return with_self[1:]  # drop self


def _get_input_tuple_from_dict(
    model: torch.nn.Module, input_dict: Dict[str, Any]
) -> Tuple:
    arg_names = _get_model_args(model)
    res = [input_dict[arg_name] for arg_name in arg_names]

    return tuple(res)


@BatchCallback.register("tensorboard_architecture_graph")
class LogArchitectureGraphtoTensorboard(BatchCallback):

    """Logs architecture graph to tensorboard using the first batch of training"""

    def __init__(self) -> None:
        # we cannot create the writer here because
        # the new design of callbacks do not pass
        # serialization directory to the callback constructors 🤷 !!
        self._writer: Optional[SummaryWriter] = None
        self._serialization_dir: Optional[Path] = None
        self.logged_once: bool = False

    @property
    def summary_writer(self) -> SummaryWriter:
        """We never return none"""

        if (self._serialization_dir is not None) and self._writer is None:
            # we are ready to create
            self._writer = SummaryWriter(self._serialization_dir)
        elif self._writer is None:
            raise RuntimeError(
                "Cannot access summary_writer without first calling "
                "check_for_serialization_dir once and making sure that _serialization_dir is set"
            )

        return self._writer

    def check_for_serialization_dir(self, trainer: GradientDescentTrainer) -> None:
        if self._serialization_dir is None:
            self._serialization_dir = Path(trainer._serialization_dir)
        elif self._serialization_dir != Path(trainer._serialization_dir):
            raise RuntimeError("Serialization dir changed unexpectedly.")

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        """Does the thing"""

        # we first need to create the writer now
        # this is brittle because it depends on the
        # trainers private attribute _serialization_dir

        if is_master:
            if not self.logged_once:
                self.check_for_serialization_dir(trainer)
                # TODO: Check if accessing trainer.model
                # works ok with DDP setup
                # the type for batch_inputs seems incorrect. It should be List[TensorDict]
                self.summary_writer.add_graph(
                    trainer.model,
                    input_to_model=_get_input_tuple_from_dict(
                        trainer.model, batch_inputs[0]  # type:ignore
                    ),
                )
                self.logged_once = True
