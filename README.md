# wandb-allennlp
Utilities and boilerplate code which allows using [Weights & Biases](https://www.wandb.com/) to tune the hypereparameters for any AllenNLP model **without a single line of extra code!**

# Features

1. Log a single run or a hyperparameter search sweep without any extra code, just using configuration files.

2. Use [Weights & Biases](https://www.wandb.com/) bayesian hyperparameter search engine + hyperband in any AllenNLP project.

3. Works with any AllenNLP version > 0.9 (including the latest 1.0.0).

4. (Coming Soon) Running parallel bayesian hyperparameter search for any AllenNLP model on a slurm managed cluster using [Weights & Biases](https://www.wandb.com/). Again without a single line of extra code.

5. (Coming Soon) Support for parameter tying to set values for interdependent hyperparameters like hidden dimension for consecutive layers. See "Advanced Use" section below.

# Status

![Tests](https://github.com/dhruvdcoder/wandb-allennlp/workflows/Tests/badge.svg)

# Quick start

## Installation

```
pip install wandb-allennlp
```

## Log a single run

1. Create your model using AllenNLP along with a *training configuration* file as you would normally do.

2. Add a trainer callback in your config file. Use one of the following based on your AllenNLP version:

For allennlp v0.9:

```
...,

trainer: {
    type: 'callback',
    callbacks: [
      ...,

      {
        type: 'log_metrics_to_wandb',
      },

      ...,
    ],
    ...,
}
...
...
```

For allennlp v1.x :

```
...

trainer: {
    epoch_callbacks: [
      ...,

      {
        type: 'log_metrics_to_wandb',
      },

      ...,
    ],
    ...,
}
...
...
```

2. Execute the following command instead of `allennlp train`:

```
wandb_allennlp --subcommand=train --config_file=model_configs/my_config.jsonnet --include-package=package_with_my_registered_classes --include-package=another_package --wandb_run_name=my_first_run --wandb_tags=any,set,of,non-unique,tags,that,identify,the,run,without,spaces

```


## Hyperparameter Search

1. Create your model using AllenNLP along with a *training configuration* file as you would normally do. For example:

```
{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
  "train_data_path": std.extVar("DATA_PATH")+"/snli_1.0_test/snli_1.0_train.jsonl",
  "validation_data_path": std.extVar("DATA_PATH")+ "/snli_1.0_test/snli_1.0_dev.jsonl",
    "model": {
            "type": "nli-seq2vec",
	    "input_size": 50,
            "hidden_size": 50,
            "rnn": "LSTM",
            "num_layers": 1,
            "bidirectional": true,
	    "projection_size": 50,
            "debug": false

    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
		"type":"callback",
		"callbacks":[
			{
				"type": "validate"
			},
			{
				"type": "checkpoint",
				"checkpointer":{
					"num_serialized_models_to_keep":1
				}
			},
			{
				"type": "track_metrics",
				"patience": 10,
				"validation_metric": "+accuracy"
			},
			{
				"type": "log_metrics_to_wandb" ###### Don't forget to include this callback.
			}
		],
		"optimizer": {
			"type": "adam",
			"lr":0.01,
			"weight_decay": 0.01
		},
		"cuda_device": -1,
		"num_epochs": 10,
		"shuffle": true
	}
}
```

2. Create a *sweep configuration* file and generate a sweep on the wandb server. For example:

```
name: nli_lstm
program: wandb_allennlp
method: bayes
## Do not for get to use the command keyword to specify the following command structure
command:
  - ${program} #omit the interpreter as we use allennlp train command directly
  - "--subcommand=train"
  - "--include-package=models" # add all packages containing your registered classes here
  - "--config_file=configs/lstm_nli.jsonnet"
  - ${args}
metric:
  name: best_validation_accuracy
  goal: maximize
parameters:
  # hyperparameters start with overrides
  # Ranges
  model.input_size:
    min: 100
    max: 500
    distribution: q_uniform
  model.hidden_size:
    min: 100
    max: 500
    distribution: q_uniform
  model.projection_size:
    min: 50
    max: 1000
    distribution: q_uniform
  model.num_layers:
    values: [1,2,3]
  model.bidirectional:
    value: "true"
  trainer.optimizer.lr:
    min: -7.0
    max: 0
    distribution: log_uniform
  trainer.optimizer.weight_decay:
    min: -12.0
    max: -5.0
    distribution: log_uniform
  model.type:
    value: nli-lstm
```

4. Set the necessary environment variables.

```
export DATA_DIR=./data
```

5. Start the search agents.

```
wandb agent <sweep_id>
```


# Advanced Use

## Parameter tying

1. Define a new jsonnet config file with source/common parameters at the top level. Set the values of these through `extVar`. Make sure to use `parseJson` to get the correct type on all of these variables.

```

local data_path = std.extVar('DATA_PATH');

\\ Special common or tying parameters
local a = std.parseJson(std.extVar('a'));
local bool_value = std.parseJson(std.extVar('bool_value'));
local int_value = std.parseJson(std.extVar('int_value'));


{
  dataset_reader: {
    ...
  },
  ...
  model: {
    type: 'parameter-tying',
    a: a,
    b: a, // a tied parameter
    bool_value: bool_value,
    bool_value_not: !bool_value, // tied parameter
    int_value: int_value,
    int_value_10: int_value + 10, // another tied parameter
    ...

  },
  ...
  trainer: {
    ...
    epoch_callbacks: ['log_metrics_to_wandb'],
  },
}
```




For detailed instructions and example see [this tutorial](http://dhruveshp.com/machinelearning/wandb-allennlp/). For an example using [allennlp-models](https://github.com/allenai/allennlp-models) see the [examples](examples) directory.
