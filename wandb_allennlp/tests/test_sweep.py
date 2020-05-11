def test_sweep(script_runner):
    ret = script_runner.run('wandb', 'agent', '--count=1', 'p285xdjw')
    assert ret.success
    assert 'wandb: Program ended successfully.' in ret.stderr
    assert 'wandb: Program failed with code 1' not in ret.stderr
