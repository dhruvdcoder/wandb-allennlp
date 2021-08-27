from typing import Tuple, List, Dict, Any, Optional
from .parser_base import WandbParserBase, read_from_env
from allennlp.commands.train import train_model_from_args
from allennlp.commands import Subcommand
import argparse
import logging
import re
import json
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path
from wandb_allennlp.config import ALLENNLP_SERIALIZATION_DIR
import shortuuid
import signal

logger = logging.getLogger(__name__)


class SigTermInterrupt(Exception):
    pass


def raise_sigterm_inturrupt(sig: int, frame: Any) -> None:
    logger.info(
        "Installed a handler for SIGTERM. It will raise SigTermInterrupt."
    )
    raise SigTermInterrupt


def generate_serialization_dir(wandb_run_id: Optional[str] = None) -> Path:
    # ref: https://github.com/wandb/client/blob/c4548d3871c4cbdd8c253e46c912c95205bbc7f6/wandb/sdk/wandb_settings.py#L740
    root_dir = Path(ALLENNLP_SERIALIZATION_DIR)
    root_dir.mkdir(parents=True, exist_ok=True)
    datetime_now: datetime = datetime.now()

    if wandb_run_id is None:
        # ref: wandb/sdk/lib/runid.py
        run_gen = shortuuid.ShortUUID(
            alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz")
        )
        wandb_run_id = run_gen.random(8)  # type: ignore[no-untyped-call]
    s = f'run-{datetime.strftime(datetime_now, "%Y%m%d_%H%M%S")}-{wandb_run_id}'

    return root_dir / s


def translate(
    hyperparams: List[str],
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    hparams = {}  #: temporary variable
    env = {}  #: params that start with env.
    all_args: List[str] = []  #: raw strings of all the unknown arguments
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

        elif len(kw_val) == 1:  # flag or positional argument
            kw_val_ = kw_val[0]
            flag = re.match("^-{1,2}(.+)", kw_val_)

            if flag:
                all_args.append(flag.group(1))
            else:  # positional arg
                # for positional arguments like param_path, we don't have to do anything
                pass

        else:
            logger.warning(
                f"{kw_val} not a know argument for allennlp train, "
                "or in --hyperparam=value form required for hyperparam overrides"
                "or a non-kwarg, "
                "or a boolean --flag"
                ". Will be ignored by train_with_wandb command."
            )

    # set the env
    # os.environ.update(env)

    return all_args, hparams, env


@Subcommand.register("train-with-wandb")
class TrainWithWandb(WandbParserBase):
    description = "Train with logging to wandb"
    help_message = (
        "Use `allennlp train_with_wandb` subcommand instead of "
        "`allennp train` to log training to wandb. "
        "It supports all the arguments present in `allennlp train`. "
        "However, the --overrides have to be specified in the `--kw value` or `--kw=value` form, "
        "where 'kw' is the parameter to override and 'value' is its value. "
        "Use the dot notation for nested parameters. "
        "For instance, {'model': {'embedder': {'type': xyz}}} can be provided as --model.embedder.type xyz"
    )
    require_run_id = False
    wandb_common_args = ["entity", "project", "notes", "group", "tags"]

    @classmethod
    def init_wandb_run(
        cls, args: argparse.Namespace
    ) -> Optional["wandb.sdk.wandb_run.Run"]:  # type: ignore

        import wandb

        wandb_args_dict = cls.get_wandb_run_args(args)

        logger.info(
            f"Early init is ON. Initializing wandb with the following args."
        )

        for k, v in wandb_args_dict.items():
            logger.info("%s  |  %s", k[:15].ljust(15), v[:15].ljust(15))

        run = wandb.init(**wandb_args_dict)
        # just use the log files and do not dynamically patch tensorboard as it messes up the
        # the global_step and breaks the normal use of wandb.log()
        # after wandb version 0.10.20
        # any call to patch either through sync_tensorboard or .patch()
        # messes up the log. So we drop it completely. Wandb somehow still
        # syncs the tensorboard log folder along with the other folders.
        # wandb.tensorboard.patch(save=True, tensorboardX=False)

        for fpath in args.wandb_files_to_save:
            self.wandb.save(  # type: ignore
                os.path.join(args.serialization_dir, fpath),
                base_path=args.serialization_dir,
                policy="live",
            )

        return run

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
        # The feeler call to parse_know_args() will throw an error.

        ######## Begin: arguments for `allennlp train`##########
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
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
        ######## End: arguments for `allennlp train`##########

        ######## Begin: Specific keyword arguments for `allennlp train_with_wandb`##########
        subparser.add_argument(
            "--early-init",
            action="store_true",
            default=False,
            help=(
                "Initialize wandb in the command processing itself."
                " The default (False) is to initialize wandb in the `on_start` method of the logging callback."
                "!!WARNING!! Early initialization of wandb can create problems when using "
                "multi-process dataloader or distributed training."
                " The only use-case for early initialization is the early population of console log in wandb UI."
            ),
        )
        subparser.add_argument(
            "--wandb-allennlp-files-to-save",
            type=str,
            action="append",
            default=[],
            help=(
                "Globs describing files to save from the allennlp serialization directory."
                "Default: ['config.json', 'out.log']"
            ),
        )
        ######## End: Specific keyword arguments for `allennlp train_with_wandb`##########

        # we will not do anything if the subcommand is not train_with_wandb
        # because otherwise parse_known_args() can throw error or show train_with_wandb's help
        # even if we are asking for --help for some other command

        if sys.argv[1] != "train-with-wandb":
            subparser.set_defaults(func=main)

            return subparser
        # Add dynamic args for overrides and env variables
        known_args, hyperparams = subparser.parse_known_args(sys.argv[2:])
        all_args, hparams_for_overrides, env_vars = translate(hyperparams)
        overrides_json = f"--overrides={json.dumps(hparams_for_overrides)}"

        # update sys.argv with the json from
        sys.argv.append(overrides_json)
        # add all hyperparams in both froms--json as well as dot notation
        # we do this so that parser_args() in the allennlp code does not throw error

        for arg in all_args:
            subparser.add_argument(f"{arg}")

        # Add the rest of the arguments of `allennlp train` that we held out due to the feeler call to parse_known_args()
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
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        # set env vars
        os.environ.update(env_vars)

        subparser.set_defaults(func=main)

        return subparser


def main(args: argparse.Namespace) -> None:
    # We keep serialization_dir and the wandb run directory serperate now.
    # We will generate a suitable seriaization-dir if not specified:
    #   1. If run_id is given either as cli argument (single run) or
    #       as environment variable (run of a sweep), we will use the same
    #       format as wandb to create a serialization-dir in ALLENNLP_SERIALIZATION_DIR
    #       as the root dir.
    #   2. If run_id cannot be obtained, we will generate a random id and treat
    #       it as run_id to generate a serialization-dir in ALLENNLP_SERIALIZATION_DIR

    # install hander for SIGTERM
    # See: https://github.com/allenai/allennlp/issues/5369
    signal.signal(signal.SIGTERM, raise_sigterm_inturrupt)

    if args.serialization_dir is None:
        logging.info(f"Set set serialization_dir as {args.serialization_dir}")
        args.serialization_dir = generate_serialization_dir(args.wandb_run_id)

    if args.early_init:
        wandb_run = TrainWithWandb.init_wandb_run(args)
    train_model_from_args(args)
