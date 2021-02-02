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
    callbacks: ['log_metrics_to_wandb'],
  },
}
