from typing import Tuple, List, Dict
from .parser_base import WandbParserBase
from allennlp.commands import Subcommand
import argparse
import wandb
import logging
import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()  # type: ignore
    run = api.run(
        f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}"
    )
    args.output_folder.mkdir(parents=True, exist_ok=True)
    pbar = tqdm.tqdm(run.files(), desc="Downloading files")

    for file_ in pbar:
        pbar.set_description(f"Downloading: {file_.name}")
        file_.download(args.output_folder, replace=args.replace)

    logger.info(f"Downloaded all files to {args.output_folder}")


@Subcommand.register("wandb_download")
class DownloadFromWandb(WandbParserBase):
    description = "Downloads all files for a run from wandb"
    help_message = (
        "Use this subcommand to perform download"
        " all the files for a particular run from wandb"
    )
    require_run_id = True
    entry_point = main

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        subparser.add_argument(
            "-o",
            "--output_folder",
            type=Path,
            required=True,
            help="Path to the output folder.",
        )
        subparser.add_argument(
            "-r",
            "--replace",
            action="store_true",
            help="Whether to overrite the contents if the files/folder exists",
        )
        subparser.set_defaults(func=main)

        return subparser
