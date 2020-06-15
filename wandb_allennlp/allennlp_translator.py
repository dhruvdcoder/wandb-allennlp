import argparse
from typing import Tuple, List, Optional, Any, Callable
import logging
import sys
import os
import re
import wandb
from pathlib import Path
from wandb_allennlp.commandline import Translator
from allennlp.__main__ import run as allennlp_run
import json

logger = logging.getLogger("allennlp_translator")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
)

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
        translator_args: argparse.Namespace,
        wandb_init: Callable,
    ) -> None:
        run = wandb_init(program, program_args, translator_args)
        sys.argv = program_args
        logger.info(f"sys.argv before running allennlp command: \n{sys.argv}")
        allennlp_run()

    def wandb_init(
        self, program: str, program_args: List[str], translator_args: argparse.Namespace
    ) -> Any:

        if translator_args.wandb_tags:
            tags = self.args.wandb_tags.split(",")
        else:
            tags = None

        wandb_run = wandb.init(
            name=translator_args.wandb_run_name,
            project=translator_args.wandb_project,
            entity=translator_args.wandb_entity,
            tags=tags,
        )
        program_args.append(
            f'--serialization-dir={Path(wandb_run.dir)/"training_dumps"}'
        )

        return wandb_run

    @classmethod
    def get_parser(
        cls, parser: argparse.ArgumentParser = None
    ) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser(description="Run wandb_allennlp")
        allennlp_group = parser.add_argument_group(
            "allennlp", "Arguments specific to allennlp train command"
        )
        allennlp_group.add_argument("--subcommand", default="train")
        allennlp_group.add_argument("--config_file", required=True, type=str)
        allennlp_group.add_argument(
            "--include-package",
            default=[],
            action="append",
            help="Same as AllenNLP's corresponding argument",
        )
        wandb_group = parser.add_argument_group(
            "wandb", "Arguments specific to wandb init"
        )

        wandb_group.add_argument(
            "--wandb_run_name",
            help="Name for the run used by wandb. Will be ignored in sweeps.",
        )
        wandb_group.add_argument("--wandb_entity")
        wandb_group.add_argument("--wandb_project")
        wandb_group.add_argument(
            "--wandb_do_not_sync_tensorboard", action="store_true")
        wandb_group.add_argument(
            "--wandb_tags", help="comma seperated list without spaces"
        )

        return parser

    def translate(
        self, translator_args: argparse.Namespace, program_args: List[str]
    ) -> Tuple[str, List[str]]:
        """At this point only hyperparameter overrides remain
        """
        hparams = {}
        # patter for starting -- or - in --key=value
        pattern = re.compile(r"-{1,2}")

        for possible_kwarg in program_args:
            kw_val = possible_kwarg.split("=")

            if len(kw_val) == 2:
                k, v = kw_val

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

        return program, args
