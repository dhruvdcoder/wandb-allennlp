from typing import Callable, Optional, List, Dict, Any
from allennlp.commands.subcommand import Subcommand
from pathlib import Path
from wandb_allennlp.utils import read_from_env
import logging
import argparse
import wandb
import os

logger = logging.getLogger(__name__)


class SetWandbEnvVar(argparse.Action):
    """Used as an action callback in argparse argument to set env vars that are read by wandb.

    Can be used like so: ::

        parser.add_argument('--wandb_entity', type=str, action=SetWandbEnvVar)
    """

    def __call__(  # type: ignore
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        assert isinstance(values, str)
        os.environ.update({self.dest.replace("-", "_").upper(): values})


class WandbParserBase(Subcommand):
    """This subcommand cannot be used directly. It is indented to
    be a common base class for all commands that use Weights & Biases.
    It as few common args for wandb.


    The way to use this in a child subcommand would be: ::

        Subcommand.register("some_subcommand")
        class SomeSubcommand(WandbParserBase):
            description = "Does some stuff"
            help_message = "Help message to do some stuff."
            entry_point=SomeCallable
            def add_arguments(self, subparser):
                subparser.add_argument("--some_important_argument", ...)
                ...
                ...
                return subparser


    """

    description: str = "A wandb base command"
    help_message: str = (
        "Most arguments starting with 'wandb_' are in one-to-one correspondance with wandb.init()."
        " See https://docs.wandb.ai/ref/run/init for reference."
    )
    require_run_id = False

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        logger.warning(
            f"add_arguments() for {self.__class__} not overriden."
            " Did you forget to add arguments?"
        )

        return subparser

    @classmethod
    def get_wandb_run_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "_".join(arg.split("_")[1:]): value
            for arg, value in vars(args).items()
            if arg.startswith("wandb_") and (value is not None)
        }

    @classmethod
    def init_wandb_run(
        cls, args: argparse.Namespace
    ) -> wandb.sdk.wandb_run.Run:
        run = wandb.init(**cls.get_wandb_run_args(args))
        # just use the log files and do not dynamically patch tensorboard as it messes up the
        # the global_step and breaks the normal use of wandb.log()
        wandb.tensorboard.patch(save=True, tensorboardX=False)

        return run  # type: ignore

    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
            self.name,
            description=self.description,
            help=self.help_message,
            conflict_handler="resolve",
        )
        subparser.add_argument(
            "--wandb-run-id",
            type=str,
            required=self.require_run_id,
            default=read_from_env("WANDB_RUN_ID"),
        )
        subparser.add_argument(
            "--wandb-entity",
            type=str,
            action=SetWandbEnvVar,
            default=read_from_env("WANDB_ENTITY"),
        )
        subparser.add_argument(
            "--wandb-project",
            type=str,
            action=SetWandbEnvVar,
            default=read_from_env("WANDB_PROJECT"),
        )
        subparser.add_argument(
            "--wandb-tags",
            type=str,
            action=SetWandbEnvVar,
            help="Comma seperated list of tags.",
        )
        subparser.add_argument("--wandb-name", action=SetWandbEnvVar, type=str)
        subparser.add_argument(
            "--wandb-group", action=SetWandbEnvVar, type=str
        )
        subparser.add_argument(
            "--wandb-job-type", action=SetWandbEnvVar, type=str
        )
        subparser.add_argument(
            "--wandb-notes", action=SetWandbEnvVar, type=str
        )
        subparser.add_argument(
            "--wandb-dir",
            type=str,
            action=SetWandbEnvVar,
            default=wandb.sdk.wandb_settings.get_wandb_dir(
                read_from_env("WANDB_DIR") or ""
            ),
        )
        # subparser.add_argument("--wandb_sync_tensorboard", action="store_true")
        subparser.add_argument(
            "--wandb-config-exclude-keys",
            type=str,
            action=SetWandbEnvVar,
            help="Comma seperated list.",
        )
        subparser.add_argument(
            "--wandb-config-include-keys",
            type=str,
            action=SetWandbEnvVar,
            help="Comma seperated list.",
        )
        subparser.add_argument(
            "--wandb-mode",
            action=SetWandbEnvVar,
            choices=["online", "offline", "disabled"],
            default="online",
        )

        subparser = self.add_arguments(subparser)

        return subparser
