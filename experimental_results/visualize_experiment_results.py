import json
import matplotlib.pyplot as plt
import numpy as np
import re

# ==============================
# 🔧 Load Results Files
# ==============================

with open('results/energyplus_variation_results.json', 'r') as f:
    energyplus_results = json.load(f)

with open('results/combined_input_variation_results.json', 'r') as f:
    combined_results = json.load(f)

with open('results/text_variation_results.json', 'r') as f:
    text_variation_results = json.load(f)

# ==============================
# 🔬 Plot 1: EnergyPlus Variation (Experiment 2)
# ==============================

# Helper: parse variable name and index
def parse_example_id(example_id):
    match = re.match(r'(ROOFR|WALLR|HVACC|HVACH)-(\d+)', example_id)
    if match:
        var_name = match.group(1)
        index = int(match.group(2))
        return var_name, index
    else:
        return None, -1

# Group by variable
target_vars = ["ROOFR", "WALLR", "HVACC", "HVACH"]
grouped = {k: [] for k in target_vars}

for entry in energyplus_results:
    var_name, index = parse_example_id(entry['example_id'])
    if var_name in target_vars:
        grouped[var_name].append({
            'index': index,
            'mean_insulation': entry['mean_insulation'],
            'std_insulation': entry['std_insulation'],
            'mean_hvac': entry['mean_hvac'],
            'std_hvac': entry['std_hvac']
        })

# Sort within groups
for var in grouped:
    grouped[var] = sorted(grouped[var], key=lambda x: x['index'])

# Plot each variable group as a separate image
for var in target_vars:
    entries = grouped[var]
    indices = [e['index'] for e in entries]
    mean_insulation = [e['mean_insulation'] for e in entries]
    std_insulation = [e['std_insulation'] for e in entries]
    mean_hvac = [e['mean_hvac'] for e in entries]
    std_hvac = [e['std_hvac'] for e in entries]

    x = np.arange(len(entries))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, mean_insulation, width, yerr=std_insulation, capsize=5, label='Insulation')
    ax.bar(x + width/2, mean_hvac, width, yerr=std_hvac, capsize=5, label='HVAC')

    ax.set_title(f'Experiment 2: {var} Variation Results')
    ax.set_xlabel(f'{var} Variation Index')
    ax.set_ylabel('Mean Label Value')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'experiment2_energyplus_variation_{var}.png')
    plt.close()

# ==============================
# 🔬 Plot 2: Combined Input Variation (Experiment 3)
# ==============================

for key, data in combined_results.items():
    conditions = list(data.keys())
    mean_insulation = [data[c]['mean_insulation'] for c in conditions]
    std_insulation = [data[c]['std_insulation'] for c in conditions]
    mean_hvac = [data[c]['mean_hvac'] for c in conditions]
    std_hvac = [data[c]['std_hvac'] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(x - width/2, mean_insulation, width, yerr=std_insulation, label='Insulation', capsize=5)
    ax.bar(x + width/2, mean_hvac, width, yerr=std_hvac, label='HVAC', capsize=5)

    ax.set_xlabel('Input Condition')
    ax.set_ylabel('Mean Label Value')
    ax.set_title(f'Experiment 3: Combined Input Variation - {key}')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'experiment3_combined_{key}.png')
    plt.close()

# ==============================
# 🔬 Plot 3: Text Variation (Experiment 1)
# ==============================

# HVAC text variation
hvac_ids = [entry['text_id'] for entry in text_variation_results['hvac']]
mean_hvac_values = [entry['mean_hvac'] for entry in text_variation_results['hvac']]
std_hvac_values = [entry['std_hvac'] for entry in text_variation_results['hvac']]

x = np.arange(len(hvac_ids))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x, mean_hvac_values, yerr=std_hvac_values, capsize=5)
ax.set_xlabel('HVAC Text Sample ID')
ax.set_ylabel('Mean HVAC Label')
ax.set_title('Experiment 1: HVAC Text Variation')
ax.set_xticks(x)
ax.set_xticklabels(hvac_ids, rotation=45)

plt.tight_layout()
plt.savefig('experiment1_text_variation_hvac.png')
plt.close()

# Insulation text variation
ins_ids = [entry['text_id'] for entry in text_variation_results['insulation']]
mean_ins_values = [entry['mean_insulation'] for entry in text_variation_results['insulation']]
std_ins_values = [entry['std_insulation'] for entry in text_variation_results['insulation']]

x = np.arange(len(ins_ids))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x, mean_ins_values, yerr=std_ins_values, capsize=5)
ax.set_xlabel('Insulation Text Sample ID')
ax.set_ylabel('Mean Insulation Label')
ax.set_title('Experiment 1: Insulation Text Variation')
ax.set_xticks(x)
ax.set_xticklabels(ins_ids, rotation=45)

plt.tight_layout()
plt.savefig('experiment1_text_variation_insulation.png')
plt.close()