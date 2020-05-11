import sys
import os
import json
from wandb_allennlp.allennlp_translator import AllenNLPSweepTranslator

os.environ['WANDB_ALLENNLP_DEBUG'] = '1'


def compare(str1, str2):
    for a, b in zip(str1, str2):
        if '--overrides' in a:
            assert json.loads(a.split('=')[1]) == json.loads(b.split('=')[1])
        else:
            assert a.strip() == b.strip()


def test_translation():
    inputs = [[
        '--subcommand=train', '--config_file=config.json', '--arg1=value1'
    ]]
    outputs = [('allennlp', [
        'allennlp', 'train', 'config.json', '--overrides={"arg1":"value1"}'
    ])]
    base_translator = AllenNLPSweepTranslator()

    for inp, out in zip(inputs, outputs):
        translated = base_translator._translate_only(inp)
        assert translated[0] == out[0]
        compare(translated[1], out[1])
