local data_path = std.extVar('DATA_PATH');
{
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
  model: {
    type: 'dummy',
    a: 50,
  },
  iterator: {
    type: 'basic',
    batch_size: 32,
  },
  trainer: {
    type: 'callback',
    callbacks: [
      {
        type: 'log_metrics_to_wandb',
      },

    ],
    optimizer: {
      type: 'adam',
      lr: 0.001,
      weight_decay: 0.0,
    },
    cuda_device: -1,
    num_epochs: 2,
    shuffle: true,
  },
}
