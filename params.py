import typing
import yaml

hparams = {}

def load_params(params_path="./hparams.yaml"):
    global hparams
    with open(params_path) as f:
        params = yaml.safe_load(f)
        for k in params:
            hparams[k] = params[k]


def set_input_size(input_size: int, nets: typing.List[str] = ["evaluator", "generator"]) -> None: 
    global hparams
    for net in nets:
        if net not in hparams:
            hparams[net] = {}
        hparams[net]["input_size"] = input_size
        
    return