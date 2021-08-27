from wandb_allennlp.training.callbacks.utils import (
    get_allennlp_major_minor_versions,
)


def test_sweep(script_runner):

    ret = script_runner.run(
        "wandb",
        "agent",
        "--count=1",
        "dhruveshpate/wandb-allennlp-wandb_allennlp_tests/k1a8rfwt",
    )
    assert ret.success
    assert "wandb: Program ended successfully." in ret.stderr
    assert "wandb: Program failed with code 1" not in ret.stderr
