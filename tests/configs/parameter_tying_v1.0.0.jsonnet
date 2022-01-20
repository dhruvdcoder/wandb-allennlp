local data_path = std.extVar('DATA_PATH');
local a = std.parseJson(std.extVar('a'));
local bool_value = std.parseJson(std.extVar('bool_value'));
local int_value = std.parseJson(std.extVar('int_value'));
local call_finish_on_end = std.parseJson(std.extVar('call_finish_on_end'));
{
  type: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'dummy',
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
    batch_size: 2,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.001,
      weight_decay: 0.0,
    },
    cuda_device: -1,
    num_epochs: 5,
    callbacks: [
      {
        type: 'wandb_allennlp',
        files_to_save: ['config.json'],
        files_to_save_at_end: ['*.tar.gz'],
        finish_on_end: call_finish_on_end,
      },
    ],
  },
}
