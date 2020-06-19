import allennlp

if int(allennlp.version._MAJOR) >= 1:
    from allennlp.__main__ import run as allennlp_run
else:
    from allennlp.run import run as allennlp_run
