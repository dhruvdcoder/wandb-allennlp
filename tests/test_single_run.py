def test_run(script_runner):
    ret = script_runner.run(
        "allennlp",
        "train-with-wandb",
        "configs/parameter_tying_v1.0.0.jsonnet",
        "--wandb-entity=dhruveshpate",
        "--wandb-project=wandb-allennlp-wandb_allennlp_tests",
        "--wandb-name=plugging_test_run",
        "--include-package=models",
        "--env.a=1.1",
        "--env.bool_value=true",
        "--env.int_value=10",
        "--model.d=1",  # keep this 1
        "--env.call_finish_on_end=false",
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
