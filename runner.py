import subprocess
import os
from pathlib import Path
import numpy as np

testing = os.name != 'posix'

limit = 10 if testing else 10 - int(
    subprocess.run(
        "qstat | grep iferreira | wc -l",
        shell=True,
        capture_output=True,
        text=True
    ).stdout.strip()
)
total = 0

def check_path_and_skip(experiment_name):
    global total, limit
    experiment_path = Path(f'experiments/{experiment_name}')

    if experiment_path.exists():
        return True
    print(total, limit)
    if total == limit: 
        print(f'Queue limit reached, exiting. \n{COUNT}/{TOTAL}, {round(COUNT*100/TOTAL, 2)}%')
        exit()

    experiment_path.mkdir(parents=True)
    total += 1
    return False


def generate_pbs_script(python_cmd, experiment_name, time):
    if testing: return

    template = Path('run.job').read_text()
    pbs_script = template.format(
        experiment_name=experiment_name,
        python_cmd=python_cmd,
        time=time
    )
    temp_file = Path("temp_pbs_script.job")
    temp_file.write_text(pbs_script)

    try:
        result = subprocess.run(['qsub', str(temp_file)], capture_output=True, text=True)
        print(f"Job submitted: {result.stdout.strip()}")
        if result.stderr:
            print(f"Errors: {result.stderr.strip()}")
    finally:
        temp_file.unlink(missing_ok=True)

def generate_python_cmd(experiment_name, noise_std, noise_target, dataset):
    output = f"python test_noise.py --noise_std {noise_std:.2f} --noise_target {noise_target} --experiment_name {experiment_name} --dataset {dataset}"
    print(output)
    return output

RUNS = 5 
NOISES = np.arange(0, 4, 0.333)
TARGETS = ['student', 'teacher', 'both']
DATASETS = ['Cifar10', 'Cifar100', 'TinyImageNet']
TIMES = {'Cifar10': '1:45:00', 'Cifar100': '2:30:00', 'TinyImageNet':'11:00:00'}

TOTAL = RUNS * len(TARGETS) * len(DATASETS) * len(NOISES)
COUNT = 0

for run in range(RUNS):
    for dataset in DATASETS:
        for noise_target in TARGETS:
            for noise_std in NOISES:
                COUNT += 1
                experiment_name = f'{dataset}/{noise_target}/std{noise_std:.2f}/{run}'
                if check_path_and_skip(experiment_name): continue
                python_cmd = generate_python_cmd(experiment_name, noise_std, noise_target, dataset)
                generate_pbs_script(python_cmd, experiment_name, TIMES[dataset])




print('All experiments are finished / queued')