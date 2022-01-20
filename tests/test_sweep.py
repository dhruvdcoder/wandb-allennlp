def test_sweep(script_runner):

    ret = script_runner.run(
        "wandb",
        "agent",
        "--count=1",
        "dhruveshpate/wandb-allennlp-wandb_allennlp_tests/fyntzj7v",
    )

    assert ret.success
    assert (
        "(success)." in ret.stderr
        or "wandb: Program ended successfully." in ret.stderr
    )
    assert not (
        "(failed 1)" in ret.stderr
        or "wandb: Program failed with code 1" in ret.stderr
    )
