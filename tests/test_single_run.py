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
    )
    assert ret.success
    assert "wandb: Program ended successfully." in ret.stderr
    assert "wandb: Program failed with code 1" not in ret.stderr
