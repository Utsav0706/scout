"""The golden-set eval: run Scout on fixed questions and score the reports.

    py -m scout.eval run                 # all cases in eval/golden.yaml
    py -m scout.eval run --tags ci       # the small CI subset

Cases use Dyno's YAML format (id, input, expected, must_call, ...), so Dyno can load the
same file later.
"""
