# wandb-allennlp


![Tests](https://github.com/dhruvdcoder/wandb-allennlp/workflows/Tests/badge.svg)


Utilities and boilerplate code which allows using [Weights & Biases](https://www.wandb.com/) to tune the hyperparameters for any AllenNLP model **without a single line of extra code!**

# What does it do?

1. Log a single run or a hyperparameter search sweep without any extra code, just using configuration files.

2. Use [Weights & Biases'](https://www.wandb.com/) bayesian hyperparameter search engine + hyperband in any AllenNLP project.



# Quick start

## Installation

```
$ pip install wandb-allennlp
$ echo wandb_allennlp >> .allennlp_plugins
```



## Log a single run

1. Create your model using AllenNLP along with a *training configuration* file as you would normally do.

2. Add a trainer callback in your config file. Use one of the following based on your AllenNLP version:


```
...,

trainer: {
    type: 'callback',
    callbacks: [
      ...,
      {
        type: 'wandb_allennlp',
        files_to_save: ['config.json'],
        files_to_save_at_end: ['*.tar.gz'],
      },
      ...,
    ],
    ...,
}
...
...
```

2. Execute the `allennlp train-with-wandb` command instead of `allennlp train`. It supports all the arguments present in `allennlp train`. However, the `--overrides` have to be specified in the `--kw value` or `--kw=value` form, where `kw` is the parameter to override and `value` is its value. Use the dot notation for nested parameters. For instance, `{'model': {'embedder': {'type': xyz}}}` can be provided as `--model.embedder.type xyz`.

```
allennlp  train-with-wandb model_configs/my_config.jsonnet --include-package=package_with_my_registered_classes --include-package=another_package --wandb-run-name=my_first_run --wandb-tags=any,set,of,non-unique,tags,that,identify,the,run,without,spaces

```


## Hyperparameter Search

1. Create your model using AllenNLP along with a *training configuration* file as you would normally do. For example:

```
local data_path = std.extVar('DATA_PATH');
local a = std.parseJson(std.extVar('a'));
local bool_value = std.parseJson(std.extVar('bool_value'));
local int_value = std.parseJson(std.extVar('int_value'));

{
  type: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'snli',
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
    },
  },
  train_data_path: data_path + '/snli_1.0_test/snli_1.0_train.jsonl',
  validation_data_path: data_path + '/snli_1.0_test/snli_1.0_dev.jsonl',
  test_data_path: data_path + '/snli_1.0_test/snli_1.0_test.jsonl',
  model: {
    type: 'parameter-tying',
    a: a,
    b: a,
    d: 0,
    bool_value: bool_value,
    bool_value_not: !bool_value,
    int_value: int_value,
    int_value_10: int_value + 10,

  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 64,
    },
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.001,
      weight_decay: 0.0,
    },
    cuda_device: -1,
    num_epochs: 2,
    callbacks: [
      {
        type: 'wandb_allennlp',
        files_to_save: ['config.json'],
        files_to_save_at_end: ['*.tar.gz'],
      },
    ],
  },
}
```

2. Create a *sweep configuration* file and generate a sweep on the wandb server. Note that the tied parameters that are accepted through environment variables are specified using the prefix `env.` in the sweep config. For example:

```
name: parameter_tying_test_console_script_v0.2.4
program: allennlp
command:
  - ${program} #omit the interpreter as we use allennlp train command directly
  - "train-with-wandb" # subcommand
  - "configs/parameter_tying_v0.2.4.jsonnet"
  - "--include-package=models" # add all packages containing your registered classes here
  - "--include-package=allennlp_models"
  - ${args}
method: bayes
metric:
  name: training_loss
  goal: minimize
parameters:
  # hyperparameters start with overrides
  # Ranges
  # Add env. to tell that it is a top level parameter
  env.a:
    min: 1
    max: 10
    distribution: uniform
  env.bool_value:
    values: [true, false]
  env.int_value:
    values: [-1, 0, 1, 10]
  model.d:
    value: 1
```
3. Create the sweep on wandb.

```
$ wandb sweep path_to_sweep.yaml
```

4. Set the other environment variables required by your jsonnet.

```
export DATA_DIR=./data
```

5. Start the search agents.

```
wandb agent <sweep_id>
```
