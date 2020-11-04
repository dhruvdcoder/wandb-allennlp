from wandb_allennlp.training.callbacks.utils import get_allennlp_major_minor_versions


def test_sweep(script_runner):
    major, minor = get_allennlp_major_minor_versions()

    if major < 1 and minor >= 9:
        ret = script_runner.run('wandb', 'agent', '--count=1',
                                'dhruveshpate/wandb_allennlp_test/p285xdjw')
    else:
        ret = script_runner.run(
            'wandb', 'agent', '--count=1',
            'dhruveshpate/wandb-allennlp-wandb_allennlp_tests/ueh5q858')

    assert ret.success
    assert 'wandb: Program ended successfully.' in ret.stderr
    assert 'wandb: Program failed with code 1' not in ret.stderr

def test_parameter_tying(script_runner):
    major, minor = get_allennlp_major_minor_versions()

    if major >=1:
        ret = script_runner.run(
            'wandb', 'agent', '--count=1',
            'dhruveshpate/wandb-allennlp-wandb_allennlp_tests/cwwdov66')

        assert ret.success
        assert 'wandb: Program ended successfully.' in ret.stderr
        assert 'wandb: Program failed with code 1' not in ret.stderr
