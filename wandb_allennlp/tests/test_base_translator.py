#import sys
#import os
#from wandb_allennlp.commandline import Translator
#
#os.environ['WANDB_ALLENNLP_DEBUG'] = '1'
#
#
# def test_translation():
#    inputs = [['"allennlp train"', 'config.json', '--arg1=value1']]
#    outputs = [('"allennlp train"',
#                ['"allennlp train"', ['config.json', '--arg1=value1']])]
#    base_translator = Translator()
#
#    for inp, out in zip(inputs, outputs):
#        assert base_translator._translate_only(inp) == out
