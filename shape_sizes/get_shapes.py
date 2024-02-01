import argparse
import torch
from vllm import LLM
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)

def print_weight_shapes(llm, layer_idx=0):
    tp_size = llm.llm_engine.parallel_config.tensor_parallel_size

    print("--- Linear Layers X * W.T. Shape of W.T:")
    proj_dict = get_projections(llm, layer_idx=layer_idx)
    
    for name in proj_dict:
        shape, tp_dim = get_shape_and_tp_dim(name, proj_dict[name])
        
        if tp_dim == "row_parallel":
            shape = [shape[0] * tp_size, shape[1]]
        elif tp_dim == "col_parallel":
            shape = [shape[0], shape[1] * tp_size]
        else:
            raise ValueError("Unknown TP Dim")

        print(f"{name}: \t{shape} \t{tp_dim}")

def get_projections(llm, layer_idx=0, attn_modules=["qkv_proj", "o_proj"], mlp_modules=["gate_up_proj", "down_proj"]):
    if llm.llm_engine.driver_worker is None:
        raise ValueError("Require TP=1")

    layer = llm.llm_engine.driver_worker.model_runner.model.model.layers[layer_idx]
    projections = {}

    for attn_module in attn_modules:
        projections[attn_module] = getattr(layer.self_attn, attn_module)
    for mlp_module in mlp_modules:
        projections[mlp_module] = getattr(layer.mlp, mlp_module)

    return projections

def get_shape_and_tp_dim(name, module):
    if not hasattr(module, "weight"):
        raise ValueError("Require Unquantized Model")
    if module.weight.dtype != torch.float16 and module.weight.dtype != torch.bfloat16:
        raise ValueError("Require Unquantized Model")
    
    if isinstance(module, RowParallelLinear):
        tp_dim = "row_parallel"
    elif isinstance(module, QKVParallelLinear) or isinstance(module, MergedColumnParallelLinear):
        tp_dim = "col_parallel"
    else:
        raise ValueError("Unknown Linear Type")

    return list(module.weight.T.shape), tp_dim

if __name__ == "__main__":
    args = parser.parse_args()

    llm = LLM(
        args.model, 
        max_model_len=1024, 
        tensor_parallel_size=torch.cuda.device_count(), 
        enforce_eager=True
    )

    print_weight_shapes(llm)