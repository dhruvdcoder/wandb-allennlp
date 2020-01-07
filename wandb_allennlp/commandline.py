#!/usr/bin/env python
"""This script is a hack which hijacks sys.argv when
the arguments are passed by wandb server. It converts
them into the format understood by the local scripts
and runs the entrypoint for all local scripts"""

import sys
from copy import deepcopy
import re
from typing import List, Iterable, Dict, Union, Optional, Callable, Tuple, Any
import json
import functools
import argparse
from collections import OrderedDict
import wandb
from pathlib import Path
import logging
import os

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

logger = logging.getLogger(__name__)


class Fixes(object):
    def _replace(self, number_str_match):  # type: ignore

        return ": " + number_str_match.group(1)  # return just the number

    def __call__(self, s):  # type: ignore

        res_copy = re.sub(r": \"([0-9.]+)\"", self._replace, s)

        return res_copy


def match_all(inp: Iterable[str],
              pattern: str = r"^--\S+=\S+$") -> Dict[str, str]:
    key_value = {}

    for k_v in inp:
        m = re.match(pattern, k_v)

        if m is None:
            raise ValueError("{} not in --key=value form".format(k_v))
        k, v = m.group(0)[2:].split('=')
        key_value[k] = v

    return key_value


def filter_level0(
        pairs: Dict[str, str]) -> Dict[str, Union[str, Dict[str, str]]]:
    """ Filter out key, value pairs for form parent.child: value"""
    result = {}

    for k, v in pairs.items():
        key_parts = k.split('.')

        if len(key_parts) > 1:  # has form parent.children
            parent = key_parts[0]  # create the new key
            subkey = '.'.join(key_parts[1:])
            parent_dict: Dict[str, str] = result.get(parent,
                                                     {})  # type: ignore
            parent_dict[subkey] = v
            result[parent] = parent_dict  # type: ignore
        else:
            result[k] = v

    return result  # type: ignore


def filter_drop_empty_values(pairs: Dict) -> Dict:
    res = {}

    for k, v in pairs.items():
        if v:
            res[k] = v

    return res


def extract_program_and_non_kwargs(everything: List[str],
                                   allowed_pattern: str = r"^[^-=]+$"
                                   ) -> Tuple[str, str]:
    words = everything
    regex = re.compile(allowed_pattern)
    filter(lambda w: not regex.match(w), words)
    prog_and_non_kwargs = []
    kwargs = []

    for word in words:
        if regex.fullmatch(word):
            prog_and_non_kwargs.append(word)
        else:
            kwargs.append(word)

    return (prog_and_non_kwargs), (kwargs)


def process_argv_args(argv,
                      fixers: Optional[List[Callable[[str], str]]] = None,
                      filters: Optional[List[Callable[[Dict], Dict]]] = None):

    if fixers is None:
        fixers = [Fixes()]

    if filters is None:
        filters = [filter_level0]
    args = deepcopy(argv)
    # first argument should be prog_name
    prog, args = extract_program_and_non_kwargs(args)
    # expected pattern
    key_value_pairs = match_all(args)
    required_form = functools.reduce(lambda res, f: f(res), filters,
                                     key_value_pairs)
    required_string_form = ' '.join([
        "--{} {}".format(k, json.dumps(v))

        if isinstance(v, dict) else "--{} {}".format(k, v)

        for k, v in required_form.items()
    ])
    fixed = functools.reduce(lambda res, f: f(res), fixers,
                             required_string_form)

    return ' '.join([prog, fixed])


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


def apply_filters_on_dict(args_dict: Dict, filters):
    filtered = functools.reduce(lambda res, f: f(res), filters, args_dict)

    return filtered


class PatterReformatter(object):
    def __init__(self, pattern=("--{}", "{}")):
        self.pattern = pattern

    def __call__(self, k, v):
        if isinstance(v, dict):
            res = (self.pattern[0].format(k), self.pattern[1].format(
                json.dumps(v)))

        else:
            res = (self.patter[0].format(k), self.pattern[1].format(v))

        return res


class PatterReformatterWithNameExceptions(PatterReformatter):
    def __init__(self,
                 pattern: Tuple[str, str] = ("--{}", "{}"),
                 exceptions: Dict[str, Callable] = None) -> None:

        if exceptions is None:
            raise ValueError("Use PatterReformatter instead")
        super().__init__(pattern)  # type: ignore
        self.exceptions = exceptions

    def __call__(self, k: Any, v: Any) -> Tuple[str, str]:

        if k in self.exceptions:
            return (self.exceptions[k])(k, v)
        else:
            return super().__call__(k, v)


def reformat_as_required(args_dict: Dict,
                         reformater=None,
                         fixers=None,
                         reorder=None) -> List[str]:

    if reformater is None:
        reformater = PatterReformatter()

    if fixers is None:
        fixers = [Fixes()]

    if reorder is not None:
        if isinstance(reorder, list):
            args_dict = functools.reduce(lambda res, f: f(res), reorder,
                                         args_dict)
        else:
            args_dict = reorder(args_dict)
    # required_string_form = [reformater(k, v) for k, v in args_dict.items()]
    required_string_form = []

    for k, v in args_dict.items():
        temp = reformater(k, v)
        required_string_form.append(temp)
    fixed = [
        functools.reduce(lambda res, f: f(res), fixers, v)

        for key_value_or_just_value_tuple in required_string_form

        for v in key_value_or_just_value_tuple
    ]

    return fixed


def init_wandb():
    pass


def sort_like(inp, ref):
    len_ref = len(ref)

    return sorted(inp, key=lambda v: ref.index(v) if v in ref else len_ref)


def sort_dict_like(inp_dict, ref_list):
    len_ref = len(ref_list)
    sorted_t = []

    for k in sorted(
            inp_dict,
            key=lambda v: ref_list.index(v) if v in ref_list else len_ref):
        sorted_t.append((k, inp_dict[k]))

    return OrderedDict(sorted_t)


class WandbToAllenNLPTranslater(object):
    def __init__(self,
                 args: Optional[Dict[Any, Any]] = None,
                 pos_args: Optional[List[str]] = None,
                 expected_wandb_args: Optional[Dict[str, str]] = None,
                 order: Optional[List[str]] = None,
                 fixed_kwargs_args: Optional[List[Tuple[str, Any]]] = None,
                 filters: List[Callable] = None,
                 fixers: List[Callable] = None,
                 keep_prog_name=True):
        self.parser = argparse.ArgumentParser()
        self.keep_prog_name = keep_prog_name

        if fixed_kwargs_args is None:
            self.fixed_kwargs_args = {}
        else:
            self.fixed_kwargs_args = fixed_kwargs_args

        if args is not None:
            for arg_name, props in args.items():
                self.parser.add_argument('--' + arg_name, **props)

        if filters is None:
            self.filters = [filter_level0, filter_drop_empty_values]
        else:
            self.filters = filters

        if fixers is None:
            self.fixers = [Fixes()]
        else:
            self.fixers = fixers

        if order is not None:
            self.reorder = lambda arr: sort_dict_like(arr, pos_args)
        else:
            self.reorder = None

        if pos_args is not None:

            def create_pos(k, v):
                return (v, )

            self.reformater = PatterReformatterWithNameExceptions(
                exceptions={name: create_pos
                            for name in pos_args})
        else:
            self.reformater = PatterReformatter()
        self.expected_wandb_args = expected_wandb_args
        self.target_args: Optional[List[str]] = None
        self.wandb_args: Optional[Dict[str, Any]] = None

    def translate(self, args: Optional[List[str]] = None) -> List[str]:
        parsed_args = create_dynamic_parser(
            known_args_parser=self.parser).parse_args(args)
        vars_dict = vars(parsed_args)
        # pop wandb args
        logger.debug("self.expected_wandb_args:{}".format(
            self.expected_wandb_args))

        if self.expected_wandb_args is not None:
            logger.debug("entered if")
            logger.debug("expected_wandb_args:{}".format(
                self.expected_wandb_args))
            self.wandb_args = {
                arg: vars_dict.pop(arg, default)

                for arg, default in self.expected_wandb_args.items()
            }
        logger.debug("wandb_args: {}".format(self.wandb_args))

        transformed = apply_filters_on_dict(vars_dict, self.filters)

        reformated = reformat_as_required(
            transformed,
            reformater=self.reformater,
            fixers=self.fixers,
            reorder=self.reorder)

        # add fixed args
        fixed_args: List[str] = []

        for k, v in self.fixed_kwargs_args:
            fixed_args += ["--{}".format(k), "{}".format(v)]
        self.target_args = reformated + fixed_args

        return self.target_args

    def translate_and_replace(self, args: Optional[List[str]] = None):
        translated_list = self.translate(args)

        if self.keep_prog_name:
            prog = [sys.argv[0]]
        else:
            prog = []

        sys.argv = prog + translated_list

        return sys.argv


def init_wandb():
    run = wandb.init()
    # add serialization dir if training

    if sys.argv[1] == 'train':
        sys.argv.append('--serialization-dir {}'.format(run.dir))


def process_wandb_arg_value(arg: str, value: Any):
    if arg == 'tags':
        return value.split(",")

    if arg == 'tensorboard':
        if value:
            wandb.tensorboard.patch(save=True, tensorboardX=False)

        return None


def setup_wandb(expected: Dict[str, Any], pos_args: List[str],
                order: List[str], fixed_kwargs_args: List[Tuple[str, str]],
                expected_wandb_args: Dict[str, Any]):

    # init wandb
    run = wandb.init()
    serialization_dir = Path(run.dir) / 'allennlp_serialization_dir'
    fixed_kwargs_args.append(('serialization-dir', str(serialization_dir)))
    translater = WandbToAllenNLPTranslater(
        expected,
        pos_args,
        expected_wandb_args=expected_wandb_args,
        order=order,
        fixed_kwargs_args=fixed_kwargs_args)
    translater.translate_and_replace()
    # use the wandb args on run

    for wandb_arg, value in translater.wandb_args.items():
        processed_val = process_wandb_arg_value(wandb_arg, value)
        logger.debug("setting {} on run object to {}".format(
            wandb_arg, processed_val))
        #setattr(run, wandb_arg, processed_val)
    # logger.debug("wandb.run object's {} attribute is {}".format(
    #        wandb_arg, getattr(wandb.run, wandb_arg)))
    #logger.debug("type of run: {}".format(type(run)))
    #logger.debug("dir of run: {}".format(dir(run)))

    return run


"""
Use as follows:
    expected = {'local_config_file': dict(type=str)}
    pos_args = ['subcommand', 'local_config_file']
    order = ['subcommand', 'local_config_file', 'serialization-dir']
    fixed_kwargs_args = [('include-package', 'datasets'),
                         ('include-package', 'models')]
    expected_wandb_args = {'tags': 'unspecified', 'tensorboard': False}

    setup_wandb(expected, pos_args, order, fixed_kwargs_args, expected_wandb_args)
    from models.run import run
    run()

"""
