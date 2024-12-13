import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_envs = [[3235.26842284954]*50,
                [3317.8367611711897]*50,
                [3349.7582080078123]*50,
                [3256.180331935489]*50,]

# File paths
files = ["./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/all_fric_avg_returns_robust.csv",
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/all_fric_avg_returns_baseline.csv",
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/dreamer_friction_returns.csv", 
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/robust_e-3_friction.csv"
         ]

files_mass = ["./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/avg_returns_robust.csv",
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/avg_returns_baseline.csv",
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/dreamer_mass_returns.csv", 
         "./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/robust_e-3_mass.csv"
         ]

colors = ["blue", "orange", "green", "red"]
labels = ["Robust e^-5",
          "Polygrad",
          "Dreamer-v3",
          "Robust e^-3"]
# Plot setup
plt.figure(figsize=(10, 6))
plt.title("Hopper", fontsize=14)
plt.xlabel("Friction", fontsize=12)
plt.ylabel("Avg Return", fontsize=12)

# Plot first dataset
for idx, file_name in enumerate(files):
    data1 = pd.read_csv(file_name)
    plt.plot(data1["Friction"], data1["Avg_Return"], label=labels[idx], color=colors[idx])
    plt.fill_between(
        data1["Friction"],
        data1["Avg_Return"] - data1["Std_Return"],
        data1["Avg_Return"] + data1["Std_Return"],
        color="blue",
        alpha=0.2,
    )

    plt.plot(data1["Friction"], training_envs[idx], "--", label=labels[idx]+" score on training env",color=colors[idx])

# Legend and grid
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.savefig("./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/ComparePlot_all_friction_dream.pdf")


plt.figure(figsize=(10, 6))
plt.title("Hopper", fontsize=14)
plt.xlabel("Mass", fontsize=12)
plt.ylabel("Avg Return", fontsize=12)

# Plot first dataset
for idx, file_name in enumerate(files_mass):
    data1 = pd.read_csv(file_name)
    plt.plot(data1["Mass"], data1["Avg_Return"], label=labels[idx], color=colors[idx])
    plt.fill_between(
        data1["Mass"],
        data1["Avg_Return"] - data1["Std_Return"],
        data1["Avg_Return"] + data1["Std_Return"],
        color="blue",
        alpha=0.2,
    )

    plt.plot(data1["Mass"], training_envs[idx], "--", label=labels[idx]+" score on training env",color=colors[idx])

# Legend and grid
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.savefig("./logs/Hopper-v3/default_2024-11-21-17:57:35_seed0/ComparePlot_all_mass_dream.pdf")