# wandb-allennlp
Utilities and boilerplate code to use wandb with allennlp

# Quick start

1. Install the package

```
pip install wandb-allennlp
```

2. Create an entrypoint file with the following content.

```
from wandb_allennlp.commandline import run

if __name__ == "__main__":
    run()
```

3. Create your model using AllenNLP along with a *training configuration* file.

4. Create a *sweep configuration* file and generate a sweep on the wandb server.

5. Set the necessary environment variables.

6. Start the search agents.

For detailed instructions and example see [this tutorial](http://dhruveshp.com/machinelearning/wandb-allennlp/).
