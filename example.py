from AgentTorch.agent_torch.core.executor import Executor
from AgentTorch.agent_torch.core.dataloader import LoadPopulation

from AgentTorch.agent_torch.models import covid
from AgentTorch.agent_torch.populations import astoria

import covid_abm

from populations import pop25001

import operator
from functools import reduce
import torch

def set_params(runner, input_string, new_value):
    tensor_func = map_and_replace_tensor(input_string)
    current_tensor = tensor_func(runner, new_value)

def map_and_replace_tensor(input_string):
    # Split the input string into its components
    parts = input_string.split('.')
    
    # Extract the relevant parts
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    arg_type = parts[4]
    var_name = parts[5]
    
    def getter_and_setter(runner, new_value=None):
        current = runner

        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)
        current_tensor = getattr(substep_function, 'calibrate_' + var_name)

        print("Current value: ", current_tensor)
        
        if new_value is not None:
            assert new_value.requires_grad == current_tensor.requires_grad
            setvar_name = 'calibrate_' + var_name
            setattr(substep_function, setvar_name, new_value)
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def setup(model, population):
    loader = LoadPopulation(population)
    simulation = Executor(model=model, pop_loader=loader)
    runner = simulation.runner

    runner.init()

    return runner 

def simulate(runner):
    num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
    print("num_steps_per_episode:", num_steps_per_episode, flush=True)

    # short run first
    steps_to_run = min(5, num_steps_per_episode)

    for t in range(steps_to_run):
        print(f"[simulate] About to run step {t}", flush=True)
        runner.step(1)
        print(f"[simulate] Finished step {t}", flush=True)

        traj = runner.state_trajectory[-1][-1]
        preds = traj['environment']['daily_infected']
        print(f"[simulate] Step {t} daily infected sum: {preds.sum().item()}", flush=True)

    loss = preds.sum()
    return loss

runner = setup(covid_abm, pop25001)
learn_params = [(name, params) for (name, params) in runner.named_parameters()]
new_tensor = torch.tensor([3.5, 4.2, 5.6, 2.0], requires_grad=True)
input_string = learn_params[0][0]

input_string = 'initializer.transition_function.0.new_transmission.learnable_args.R2'

params_dict = {input_string: new_tensor}
runner._set_parameters(params_dict)

print(simulate(runner))

# set_params(runner, input_string, new_tensor)

'''
Tasks to do:
1. Custom population size 
2. Init Infections
3. Set parameters
4. Visualize values
'''

