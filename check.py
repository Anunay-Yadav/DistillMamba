
from accelerate import Accelerator
import wandb
# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb", )

accelerator.init_trackers(
    project_name="mamba_distill",
    init_kwargs={"wandb": {"name": "new2"}}
    )

import json

# Load JSON from a file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

json = load_json("output/mamba_init_first8/lm-eval-results.json")
my_table = wandb.Table(columns=["task", "acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"]
                       )
columns = ["acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"]

results = dict(json)['results']
for task in results:
    results_per_task_dict = {}
    for col in columns: 
        results_per_task_dict[col] = 0
        if col in results[task]:
            results_per_task_dict[col] = results[task][col]
    my_table.add_data(task, results_per_task_dict[columns[0]], results_per_task_dict[columns[1]], results_per_task_dict[columns[2]], results_per_task_dict[columns[3]])
accelerator.log({"lm-eval-results": my_table})