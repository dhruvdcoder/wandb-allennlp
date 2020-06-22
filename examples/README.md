# Examples

## Running Hyperparameter search for AllenNLP models from allennlp-models repository.

1. Install the models repository (use virtual environment or conda)

```
pip install allennlp-models
```

2. Create a model using config file. See [training_configs/pair_classification/decomposable_attention.jsonnet](examples/training_configs/pair_classification/decomposable_attention.jsonnet) for complete example config.

**Note: Use the following callback specification format for allennlp v0.9:**

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

**and the following for allennlp v1.x :**
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

3. Create a sweep config file. See [sweep_configs/pair_classification/decomposable_attention.yaml](examples/sweep_configs/pair_classification/decomposable_attention.yaml) for the example config.

4. Create the sweep.

```
wandb sweep sweep_configs/pair_classification/decomposable_attention.yaml
```

5. Start the agent(s). Run the following command on multiple machines for parallelizing the search.

```
wandb agent <sweep-id>
```

6. Look at the results! Results for the presented examples can be found [here](https://app.wandb.ai/dhruveshpate/wandb_allennlp_models_demo/sweeps/vwwu3sa0).


