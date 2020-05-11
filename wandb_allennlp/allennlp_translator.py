import argparse
from typing import Tuple, List, Optional, Any
import logging
import sys
import os
import re
import wandb
from pathlib import Path
from wandb_allennlp.commandline import Translator
from allennlp.run import run as allennlp_run
import json
logger = logging.getLogger('allennlp_translator')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

logger.addHandler(handler)

if os.environ.get("WANDB_ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logger.setLevel(LEVEL)


class AllenNLPCommandLauncherMixin:
    def wandb_init(self, program: str, program_args: List[str],
                   translator_args: argparse.Namespace) -> Any:

        if translator_args.wandb_tags:
            tags = self.args.wandb_tags.split(',')
        else:
            tags = None

        wandb_run = wandb.init(
            name=translator_args.wandb_run_name,
            project=translator_args.wandb_project,
            entity=translator_args.wandb_entity,
            tags=tags,
        )
        program_args.append(
            f'--serialization-dir={Path(wandb_run.dir)/"training_dumps"}')

        return wandb_run

    def _launch_process(self, program: str, program_args: List[str],
                        translator_args: argparse.Namespace) -> None:
        run = self.wandb_init(program, program_args, translator_args)
        sys.argv = program_args
        logger.info(f"sys.argv before running allennlp command: \n{sys.argv}")
        allennlp_run()


class AllenNLPSweepTranslator(AllenNLPCommandLauncherMixin, Translator):
    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser(description="Run wandb_allennlp")
        parser.add_argument('--subcommand', default='train')
        parser.add_argument('--config_file', required=True, type=str)
        parser.add_argument(
            '--include-package',
            default=[],
            action='append',
            help='Same as AllenNLP\'s corresponding argument')
        parser.add_argument(
            '--wandb_run_name', help='Name for the run used by wandb')
        parser.add_argument('--wandb_entity')
        parser.add_argument('--wandb_project')
        parser.add_argument('--wandb_sync_tensorboard')
        parser.add_argument(
            '--wandb_tags', help='comma seperated list without spaces')

        return parser

    def translate(self, translator_args: argparse.Namespace,
                  program_args: List[str]) -> Tuple[str, List[str]]:
        """At this point only hyperparameter overrides remain
        """
        hparams = {}
        pattern = re.compile(
            r'-{1,2}')  # patter for starting -- or - in --key=value

        for possible_kwarg in program_args:
            kw_val = possible_kwarg.split('=')

            if len(kw_val) == 2:
                k, v = kw_val

                hparams[pattern.sub('', k)] = v
            else:
                logger.warning(
                    f"{kw_val} not in --key=value form. Will be ignored.")
        program = 'allennlp'
        args = [
            program, translator_args.subcommand, translator_args.config_file,
            f"--overrides={json.dumps(hparams)}",
            "--include-package=wandb_allennlp"
        ]

        for package in translator_args.include_package:
            args.append(f'--include-package={package}')

        return program, args
