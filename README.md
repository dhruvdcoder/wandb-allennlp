# wandb-allennlp
Utilities and boilerplate code which allows using wandb to tune the hypereparameters for any AllenNLP model without a single line of extra code!

# Status

![Tests](https://github.com/dhruvdcoder/wandb-allennlp/workflows/Tests/badge.svg)

# Quick start

1. Install the package

```
pip install wandb-allennlp
```

2. Create your model using AllenNLP along with a *training configuration* file. For example:

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

4. Create a *sweep configuration* file and generate a sweep on the wandb server. For example:

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

5. Set the necessary environment variables.

```
export DATA_DIR=./data
```

6. Start the search agents.

```
wandb agent <sweep_id>
```

For detailed instructions and example see [this tutorial](http://dhruveshp.com/machinelearning/wandb-allennlp/).
