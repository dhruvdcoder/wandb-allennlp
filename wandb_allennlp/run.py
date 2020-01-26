from wandb_allennlp.commandline import setup_wandb
from allennlp.run import run
import os

# arguments which SHOULD be present
_expected = {'local_config_file': dict(type=str)}

# All the arguments from wandb server will be
# named arguments. Following specifies which of these
# to convert to positional arguments while passing them to allennlp
# specify the arguments which are to be converted
# example below will create:
# $ allennlp <subcommand> <local_config_file> ...
# concretely:
# $ allennlp train configs/lstm_nli.jsonnet
_pos_args = ['subcommand', 'local_config_file']

# if arguments have to follow a strict order, specify it here
# Most likely, positions arguments should be required to follow a specific order
_order = ['subcommand', 'local_config_file', 'serialization-dir']

# These are arguments which will not be sent by the wandb server
# but allennlp always needs these
_fixed_kwargs_args = None


def include_packages() -> None:
    packages: str = os.environ.get('include_package')

    if packages is not None:
        if _fixed_kwargs_args is None:
            _fixed_kwargs_args = []

        for package in packages.split(','):
            _fixed_kwargs_args.append(('include-package', package.strip()))


_expected_wandb_args = {'tags': 'unspecified', 'tensorboard': False}


def wandb_allennlp_run():
    setup_wandb(_expected, _pos_args, _order, _fixed_kwargs_args,
                _expected_wandb_args)

    run()


if __name__ == "__main__":
    wandb_allennlp_run()
