from typing import Tuple, List, Dict, Any
from .parser_base import WandbParserBase
from allennlp.commands.train import train_model_from_args
from allennlp.commands import Subcommand
import argparse
import wandb
import logging
import re
import json
import yaml
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def translate(
    hyperparams: List[str],
) -> Tuple[List[str], str, Dict[str, Any]]:
    hparams = {}  #: temporary variable
    env = {}  #: params that start with env.
    all_args: List[str] = []  #: Names of all the unknown arguments
    # patter for starting -- or - in --key=value
    pattern = re.compile(r"-{1,2}")

    for possible_kwarg in hyperparams:
        kw_val = possible_kwarg.split("=")

        if len(kw_val) > 2:
            raise ValueError(f"{possible_kwarg} not in valid form.")

        elif len(kw_val) == 2:
            k, v = kw_val
            all_args.append(k)
            # pass through yaml.load to handle
            # booleans, ints and floats correctly
            # yaml.load with output correct python types
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(  # type: ignore
                "tag:yaml.org,2002:float",
                re.compile(
                    """^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$""",
                    re.X,
                ),
                list("-+0123456789."),
            )
            v = yaml.load(v, Loader=loader)

            if k.startswith("--env.") or k.startswith("-env."):
                # split on . and remove the "--env." in the begining
                # use json dumps to convert the python type (int, float, bool)
                # to string which can be understood by a json reader
                # the environment variables have to be stored as string
                env[".".join(k.split(".")[1:])] = json.dumps(v)
            else:
                hparams[pattern.sub("", k)] = v

        elif len(kw_val) == 1:
            all_args.append(kw_val) # possible non-kwarg or store_true flag
        else:
            logger.warning(
                f"{kw_val} not a know argument for allennlp train, "
                "or in --hyperparam=value form required for hyperparam overrides"
                "or a non-kwarg, "
                "or a boolean --flag"
                ". Will be ignored by train_with_wandb command."
            )

    hyperparams_json = f"--overrides={json.dumps(hparams)}"

    # set the env
    # os.environ.update(env)

    return all_args, hyperparams_json, env


@Subcommand.register("train_with_wandb")
class TrainWithWandb(WandbParserBase):
    description = "Train with logging to wandb"
    help_message = (
        "Use `allennlp train_with_wandb` subcommand instead of "
        "`allennp train` to log training to wandb"
    )
    require_run_id = False

    @classmethod
    def init_wandb_run(
        cls, args: argparse.Namespace
    ) -> wandb.sdk.wandb_run.Run:
        wandb_args_dict = cls.get_wandb_run_args(args)

        if (args.serialization_dir is not None) and (
            wandb_args_dict.get("dir") is not None
        ):
            raise ValueError(
                f"Cannot specify both --serialization_dir and --wandb_dir "
                "at the same time."
            )

        if args.serialization_dir:
            wandb_args_dict["dir"] = args.serialization_dir
            serialization_dir_configured = True
        run = wandb.init(**wandb_args_dict)
        # just use the log files and do not dynamically patch tensorboard as it messes up the
        # the global_step and breaks the normal use of wandb.log()
        # after wandb version 0.10.20
        # any call to patch either through sync_tensorboard or .patch()
        # messes up the log. So we drop it completely. Wandb somehow still
        # syncs the tensorboard log folder along with the other folders.
        # wandb.tensorboard.patch(save=True, tensorboardX=False)
        args.serialization_dir = f"{run.dir}/training_dump"  # type:ignore

        return run  # type:ignore

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        # we use the same args as the allennlp train command
        # except the --overrides
        # and param_path because
        # overrides is something we will create
        # and param_path is not a kwarg and hence is always required
        # We cannot have a compulsory arg here because if we do and 
        # we are not trying to call train_with_wandb but some other command
        # The call feeler call to parse_know_args() will throw an error.
        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=False,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "--node-rank",
            type=int,
            default=0,
            help="rank of this node in the distributed setup",
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )

        # Add dynamic args for overrides and env variables
        known_args, hyperparams = subparser.parse_known_args(sys.argv[2:])
        all_args, overrides_json, env_vars = translate(hyperparams)
        # update sys.argv with the json from
        sys.argv.append(overrides_json)
        # add all hyperparams in both froms--json as well as dot notation
        # we do this so that parser_args() in the allennlp code does not throw error

        for arg in all_args:
            subparser.add_argument(f"{arg}")
        # set env vars
        os.environ.update(env_vars)

        subparser.set_defaults(func=main)

        return subparser


def main(args: argparse.Namespace) -> None:
    wandb_run = TrainWithWandb.init_wandb_run(args)
    train_model_from_args(args)
