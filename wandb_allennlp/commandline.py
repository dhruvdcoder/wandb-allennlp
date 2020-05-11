"""Utilities to translate parameters passed by wandb server to
a target format (currently supporting AllenNLP commands only)
"""
import argparse
from typing import Tuple, List, Optional
import logging
import sys
import os

logger = logging.getLogger('translator')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

logger.addHandler(handler)

if os.environ.get("WANDB_ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logger.setLevel(LEVEL)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    return parser


def create_dynamic_parser(
        args: Optional[List[str]] = None,
        known_args_parser: Optional[argparse.ArgumentParser] = None):

    if known_args_parser is None:
        known_args_parser = argparse.ArgumentParser()

    if known_args_parser is not None:
        known_args, unknown_args = known_args_parser.parse_known_args(args)

    for unknown_arg in unknown_args:
        # expect them to be like --name=value
        try:
            name, value = unknown_arg.split('=')
        except Exception as e:
            raise ValueError(
                "{} no in --key=value form".format(unknown_arg)) from e

        if name.startswith(("-", "--")):
            known_args_parser.add_argument(name)
        else:
            raise ValueError("{} no in --key=value form".format(unknown_arg))

    return known_args_parser


class ExecvpLauncherMixin:
    def _launch_process(self, program: str, program_args: List[str]) -> None:
        sys.stdout.flush()
        os.execvp(program, program_args)


class Translator:
    def __init__(self):
        self.parser = self.get_parser()
        self.args = None

    @classmethod
    def get_parser(cls, parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            '--interpreter',
            type=str,
            help="Path to or name of interpreter "
            "like python3 etc."
            "If not passed, the target program will not "
            "be invoked with an explicit interpreter")
        # Add any arguments you want to pass to the translater here using
        # parser.add_argument()
        # for instance if you want to do something like:
        # python -m translator --some_translater_arg=a train.py --train_arg1=b
        # then add --some_translator_arg here.
        parser.add_argument("train_script_or_command")

        return parser

    def _parse_args(self,
                    args: List[str]) -> Tuple[argparse.Namespace, List[str]]:

        translator_args, program_args = self.parser.parse_known_args(args)
        self.args = translator_args

        return translator_args, program_args

    def translate(self, translator_args: argparse.Namespace,
                  program_args: List[str]) -> Tuple[str, List[str]]:
        """Return prog and translated args"""
        raise NotImplementedError

    def _launch_process(self, program: str, program_args: List[str]) -> None:
        pass

    def _translate_only(self, cmd_args: List[str]) -> Tuple[str, List[str]]:
        translator_args, program_args = self._parse_args(cmd_args)
        prog, args = self.translate(translator_args, program_args)

        return prog, args

    def __call__(self, cmd_args: List[str] = None):

        self._launch_process(*self._translate_only(cmd_args), self.args)
