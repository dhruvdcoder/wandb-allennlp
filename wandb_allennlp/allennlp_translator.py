import argparse
from typing import Tuple, List, Optional, Any, Callable
import logging
import sys
import os
import re
import wandb
from pathlib import Path
from wandb_allennlp.commandline import Translator
from .versioned import allennlp_run
import json
import yaml

logger = logging.getLogger("allennlp_translator")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

logger.addHandler(handler)

if os.environ.get("WANDB_ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logger.setLevel(LEVEL)


class WandbAllenNLPTranslator(Translator):
    """Currently specialized only for allennlp train command"""

    def _launch_process(
            self,
            program: str,
            program_args: List[str],
            translator_args: Optional[argparse.Namespace],
            wandb_init: Callable,
    ) -> None:
        run = wandb_init(program, program_args, translator_args)
        sys.argv = program_args
        logger.info(f"sys.argv before running allennlp command: \n{sys.argv}")
        allennlp_run()

    def wandb_init(self, program: str, program_args: List[str],
                   translator_args: argparse.Namespace) -> Any:

        if translator_args is not None:
            if translator_args.wandb_tags:
                tags = translator_args.wandb_tags.split(",")
            else:
                tags = None

            wandb_run = wandb.init(
                name=translator_args.wandb_run_name,
                project=translator_args.wandb_project,
                entity=translator_args.wandb_entity,
                tags=tags,
                sync_tensorboard=False)
        else:
            wandb_run = wandb.init()
        # just use the log files and do not dynamically patch tensorboard as it messes up the
        # the global_step and breaks the normal use of wandb.log()
        wandb.tensorboard.patch(save=True, tensorboardX=False)
        program_args.append(
            f'--serialization-dir={Path(wandb_run.dir)/"training_dumps"}')

        return wandb_run

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser = None
                   ) -> argparse.ArgumentParser:

        if parser is None:
            parser = argparse.ArgumentParser(description="Run wandb_allennlp")
        allennlp_group = parser.add_argument_group(
            "allennlp", "Arguments specific to allennlp train command")
        allennlp_group.add_argument("--subcommand", default="train")
        allennlp_group.add_argument("--config_file", required=True, type=str)
        allennlp_group.add_argument(
            "--include-package",
            default=[],
            action="append",
            help="Same as AllenNLP's corresponding argument",
        )
        wandb_group = parser.add_argument_group(
            "wandb", "Arguments specific to wandb init")

        wandb_group.add_argument(
            "--wandb_run_name",
            help="Name for the run used by wandb. Will be ignored in sweeps.",
        )
        wandb_group.add_argument("--wandb_entity")
        wandb_group.add_argument("--wandb_project")
        wandb_group.add_argument(
            "--wandb_do_not_sync_tensorboard", action="store_true")
        wandb_group.add_argument(
            "--wandb_tags", help="comma seperated list without spaces")

        return parser

    def translate(self, translator_args: argparse.Namespace,
                  program_args: List[str]) -> Tuple[str, List[str]]:
        """At this point only hyperparameter overrides remain
        """
        hparams = {}
        env = {} # params that start with env.
        # patter for starting -- or - in --key=value
        pattern = re.compile(r"-{1,2}")

        for possible_kwarg in program_args:
            kw_val = possible_kwarg.split("=")

            if len(kw_val) == 2:
                k, v = kw_val

                # pass through yaml.load to handle
                # booleans, ints and floats correctly
                # yaml.load with output correct python types
                v = yaml.load(v)

                if k.startswith("--env.") or k.startswith("-env."):
                    # split on . and remove the "--env." in the begining
                    # use json dumps to convert the python type (int, float, bool)
                    # to string which can be understood by a json reader
                    # the environment variables have to be stored as string
                    env[".".join(k.split(".")[1:])] = json.dumps(v)
                else:
                    hparams[pattern.sub("", k)] = v

            else:
                logger.warning(
                    f"{kw_val} not in --key=value form. Will be ignored.")

        program = "allennlp"
        args = [
            program,
            translator_args.subcommand,
            translator_args.config_file,
            f"--overrides={json.dumps(hparams)}",
            "--include-package=wandb_allennlp",
        ]

        for package in translator_args.include_package:
            args.append(f"--include-package={package}")

        # set the env
        os.environ.update(env)

        return program, args
