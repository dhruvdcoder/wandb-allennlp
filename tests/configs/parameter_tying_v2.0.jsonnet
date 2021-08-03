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
