def test_run_fail_and_dont_close(script_runner):
    """
    The model training will fail after one epoch.
    The method `wandb.close()` will not be called in the
    `close()` function of the wandb callback.
    Hence, the fact that the run has failed should propogate to wandb.
    """
    ret = script_runner.run(
        "allennlp",
        "train-with-wandb",
        "configs/parameter_tying_v1.0.0.jsonnet",
        "--wandb-entity=dhruveshpate",
        "--wandb-project=wandb-allennlp-wandb_allennlp_tests",
        "--wandb-name=plugging_test__test_run_fail_and_dont_close",
        "--include-package=models",
        "--env.a=1.1",
        "--env.bool_value=true",
        "--env.int_value=10",
        "--model.type=parameter-tying-failing",
        "--model.d=1",  # keep this 1
        "--env.call_finish_on_end=false",
    )
    assert not ret.success
    assert "wandb: Program ended successfully." not in ret.stderr
    assert "wandb: Program failed with code 1" in ret.stderr


def test_run_fail_and_close(script_runner):
    """
    The model training will fail after one epoch.
    The method `wandb.close()` will be called in the
    `close()` function of the wandb callback.
    Hence, the fact that the run has failed should not propogate to wandb.
    """
    ret = script_runner.run(
        "allennlp",
        "train-with-wandb",
        "configs/parameter_tying_v1.0.0.jsonnet",
        "--wandb-entity=dhruveshpate",
        "--wandb-project=wandb-allennlp-wandb_allennlp_tests",
        "--wandb-name=plugging_test__test_run_fail_and_close",
        "--include-package=models",
        "--env.a=1.1",
        "--env.bool_value=true",
        "--env.int_value=10",
        "--model.type=parameter-tying-failing",
        "--model.d=1",  # keep this 1
        "--env.call_finish_on_end=true",
    )
    assert not ret.success
    assert "wandb: Program ended successfully." in ret.stderr
    assert "wandb: Program failed with code 1" not in ret.stderr
