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
				"type": "log_metrics_to_wandb"
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
