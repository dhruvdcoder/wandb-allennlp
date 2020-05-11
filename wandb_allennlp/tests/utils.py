import pytest


@pytest.fixture
def base_translator():
    from wandb_allennlp.commandline import Translator

    return Translator()
