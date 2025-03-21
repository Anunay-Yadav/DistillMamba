import os

from safetensors import safe_open
import torch
def load_safetensors_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    safetensors_dict[key] = f.get_tensor(key)
    return safetensors_dict

def construct_layer_dict(safetensors_dict, num_hidden_layers):
    layer_dict = {}
    is_mamba_layer = [False for _ in range(num_hidden_layers)]
    prefix = "model.layers."
    for full_key, tensor in safetensors_dict.items():
        if full_key.startswith(prefix):
            parts = full_key[len(prefix):].split('.', 1)
            layer_id = int(parts[0])
            param_name = parts[1]
            if layer_id not in layer_dict:
                layer_dict[layer_id] = {}
            if "mamba" in param_name:
                is_mamba_layer[layer_id] = True
            layer_dict[layer_id][param_name] = tensor
    return layer_dict, is_mamba_layer
def load_dataset(datasets_path):
    data = []
    label = []
    for dataset_path in datasets_path:
        temp_dataset = torch.load(dataset_path)
        input_ids, labels = temp_dataset.tensors
        data.append(input_ids)
        label.append(labels)
    return torch.cat(data, dim = 0), torch.cat(label, dim = 0)

def log_lm_eval_results(file_path, accelerator):
    import json

# Load JSON from a file
import json
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return dict(data)['results']

def log_lm_eval_results(file_path, accelerator):
    import wandb
    results_dict = load_json(file_path + "lm-eval-results.json")
    
    my_table = wandb.Table(columns=["task", "acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"]
                        )
    columns = ["acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"]

    for task in results_dict:
        results_per_task_dict = {}
        for col in columns: 
            results_per_task_dict[col] = 0
            if col in results_dict[task]:
                results_per_task_dict[col] = results_dict[task][col]
        my_table.add_data(task, results_per_task_dict[columns[0]], results_per_task_dict[columns[1]], results_per_task_dict[columns[2]], results_per_task_dict[columns[3]])
    accelerator.log({"lm-eval-results": my_table})