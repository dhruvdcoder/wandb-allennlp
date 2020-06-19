#!/usr/bin/env python
import os
import sys
from wandb_allennlp.allennlp_translator import WandbAllenNLPTranslator

sys.path.insert(0, os.path.dirname(
    os.path.abspath(os.path.join(__file__, os.pardir))))


def run() -> None:
    translator = WandbAllenNLPTranslator()
    translator()


if __name__ == "__main__":
    run()
