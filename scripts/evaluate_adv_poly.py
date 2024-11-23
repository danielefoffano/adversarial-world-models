from polygrad.utils.evaluation import evaluate_policy
from polygrad.utils.envs import create_env
import torch
import numpy as np
import polygrad.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    config: str = "config.simple_maze"


args = Parser().parse_args()

expl_env = create_env(args.env_name, args.suite)
eval_env = create_env(args.env_name, args.suite)
random_episodes = utils.rl.random_exploration(args.n_prefill_steps, expl_env)

print("Seed", args.seed)
utils.set_all_seeds(args.seed)
all_metrics = []
# load all config params

configs = utils.create_configs(args, eval_env)
if configs["render_config"] is not None:
    renderer = configs["render_config"]()
else:
    renderer = None
model = configs["model_config"]()
diffusion = configs["diffusion_config"](model)
value_model = configs["value_model_config"]()
value_diffusion = configs["value_diffusion_config"](value_model)
dataset = configs["dataset_config"](random_episodes)
diffusion_trainer = configs["trainer_config"](diffusion, dataset, eval_env, value_diffusion, renderer)
ac = configs["ac_config"](normalizer=dataset.normalizer)
agent = configs["agent_config"](
    diffusion_model=diffusion_trainer.ema_model,
    actor_critic=ac,
    dataset=dataset,
    env=eval_env,
    renderer=renderer,
    value_model = diffusion_trainer.ema_model_value #DF-CHANGE
)

path = "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/"
step = 999999
agent.load(path, step)

masses = np.linspace(2.0, 4.7, 20)
for mass in masses:
    eval_env.sim.model.body_mass[1] = mass
    eval_metrics = evaluate_policy(
                ac.forward_actor,
                eval_env,
                device,
                step,
                dataset,
                use_mean=True,
                n_episodes=100,
                renderer=renderer,
            )
    all_metrics.append(eval_metrics)
    print(f"Metrics for mass {eval_env.sim.model.body_mass[1]}")
    print(eval_metrics)

print("done")

import matplotlib.pyplot as plt

means = [metric["avg_return"] for metric in all_metrics]
min_ret = [metric["min_return"] for metric in all_metrics]
max_ret = [metric["max_return"] for metric in all_metrics]

fig, ax = plt.subplots()
ax.plot(masses, means)
ax.set(xlabel="mass", ylabel="cumulative return", title="Hopper-v3")
ax.grid()
fig.savefig("./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/HopperPlot.pdf")