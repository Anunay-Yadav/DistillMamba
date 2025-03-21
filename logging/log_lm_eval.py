
from accelerate import Accelerator
import wandb
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
# print(parent)
sys.path.append(parent)

from util import load_json
import json

# Load JSON from a file

import os
def process_files_in_directory(root_dir):
    results_for_each_model = {}
    for subdir, _, files in os.walk(root_dir):  # Walk through each subdirectory
        for file in files:
            if file == "lm-eval-results.json":  # Check if the file is abc.json
                file_path = os.path.join(subdir, file)  # Construct full file path
                
                if os.path.exists(file_path):  # Check if file exists
                    data = load_json(file_path)
                    results_for_each_model[subdir.split("/")[1]] = data
                else:
                    print(f"File not found: {file_path}")
    return results_for_each_model


def log_wandb_table(results_for_each_model, columns):
    lm_eval_results_table = wandb.Table(columns=["task", "model_name"] + columns)
    data = []
    for model in results_for_each_model:
        results_dict = results_for_each_model[model]
        for task in results_dict:
            results_per_task_dict = {}
            for col in columns: 
                results_per_task_dict[col] = 0
                if col in results_dict[task]:
                    results_per_task_dict[col] = results_dict[task][col]
            row = []
            row.append(task)
            row.append(model)
            for col in columns:
                row.append(results_per_task_dict[col])
            data.append(row)
    
    data.sort(key=lambda x: -x[2])
    data.sort(key=lambda x: x[0])
    for row in data:
        lm_eval_results_table.add_data(*row)
    return lm_eval_results_table
def log_wandb_task_wise_table(results_for_each_model, columns):
    tasks = set()
    for model in results_for_each_model:
        results_dict = results_for_each_model[model]
        for task in results_dict:
            tasks.add(task)
    
    for task in tasks:
        task_table = wandb.Table(columns=["model_name"] + columns)
        data = []
        for model in results_for_each_model:
            results_dict = results_for_each_model[model]
            results_per_task_dict = {}
            for col in columns: 
                results_per_task_dict[col] = 0
                if col in results_dict[task]:
                    results_per_task_dict[col] = results_dict[task][col]
            row = []
            row.append(model)
            for col in columns:
                row.append(results_per_task_dict[col])
            data.append(row)
        data.sort(key=lambda x: -x[1])
        for row in data:
            task_table.add_data(*row)
        wandb.log({task : task_table})

if __name__ == '__main__':
    is_new = False 
    if(is_new): #change for first time

        run = wandb.init(project="mamba_distill", name="benchmark_metrics")
        with open(".run_id.txt", "w") as file:
            file.write(run.id)
    else:
        # read run_id of the previous run
        with open(".run_id.txt") as file:
            run_id = file.readline()

        run = wandb.init(id=run_id, resume=True, project="mamba_distill",  name="benchmark_metrics")

    columns = ["acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"]
    results_for_each_model = process_files_in_directory("output/")
    lm_eval_results_table = log_wandb_table( results_for_each_model, columns)
    wandb.log({"lm-hairness-eval-metrics" : lm_eval_results_table})
    log_wandb_task_wise_table(results_for_each_model, columns)
    wandb.finish()