import json
import matplotlib.pyplot as plt
import numpy as np
import re


def main(results_dir, include_weight_optimization=False):
    # ==============================
    # 🔧 Load Results Files
    # ==============================

    with open(results_dir / 'energyplus_results.json', 'r') as f:
        energyplus_results = json.load(f)

    with open(results_dir / 'combined_results.json', 'r') as f:
        combined_results = json.load(f)

    with open(results_dir / 'text_results.json', 'r') as f:
        text_variation_results = json.load(f)

    with open(results_dir / 'optimize_weights_results.json', 'r') as f:
        optimize_weights_results = json.load(f)

    # ==============================
    # 🔬 Plot 1: EnergyPlus Variation (Experiment 2)
    # ==============================

    def parse_example_id(example_id):
        match = re.match(r'(\w+)-(\d+)', example_id)
        if match:
            var_name = match.group(1)
            index = int(match.group(2))
            return var_name, index
        else:
            return example_id, -1

    grouped = {}
    for entry in energyplus_results:
        var_name, index = parse_example_id(entry['example_id'])
        if var_name not in grouped:
            grouped[var_name] = []
        grouped[var_name].append({
            'index': index,
            'mean_insulation': entry['mean_insulation'],
            'std_insulation': entry['std_insulation'],
            'mean_hvac': entry['mean_hvac'],
            'std_hvac': entry['std_hvac'],
            'example_id': entry['example_id']
        })

    for var in grouped:
        grouped[var] = sorted(grouped[var], key=lambda x: x['index'])

    for var, entries in grouped.items():
        indices = [e['index'] for e in entries]
        mean_insulation = [e['mean_insulation'] for e in entries]
        std_insulation = [e['std_insulation'] for e in entries]
        mean_hvac = [e['mean_hvac'] for e in entries]
        std_hvac = [e['std_hvac'] for e in entries]

        x = np.arange(len(entries))
        width = 0.3

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, mean_insulation, width, yerr=std_insulation, capsize=5, label='Insulation')
        ax.bar(x + width / 2, mean_hvac, width, yerr=std_hvac, capsize=5, label='HVAC')

        ax.set_xlabel(f'{var} Variation Index')
        ax.set_ylabel('Mean Label Value')
        ax.set_title(f'Experiment 2: {var} Variation Results')
        ax.set_xticks(x)
        ax.set_xticklabels(indices)
        ax.legend()

        plt.tight_layout()
        plt.savefig(results_dir / f'experiment2_energyplus_variation_{var}.png')
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

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, mean_insulation, width, yerr=std_insulation, label='Insulation', capsize=5)
        ax.bar(x + width / 2, mean_hvac, width, yerr=std_hvac, label='HVAC', capsize=5)

        ax.set_xlabel('Input Condition')
        ax.set_ylabel('Mean Label Value')
        ax.set_title(f'Experiment 3: Combined Input Variation - {key}')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(results_dir / f'experiment3_combined_{key}.png')
        plt.close()

    # ==============================
    # 🔬 Plot 3: Text Variation (Experiment 1)
    # ==============================

    hvac_ids = [entry['text_id'] for entry in text_variation_results['hvac']]
    mean_hvac_values = [entry['mean_hvac'] for entry in text_variation_results['hvac']]
    std_hvac_values = [entry['std_hvac'] for entry in text_variation_results['hvac']]

    x = np.arange(len(hvac_ids))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, mean_hvac_values, yerr=std_hvac_values, capsize=5)
    ax.set_xlabel('HVAC Text Sample ID')
    ax.set_ylabel('Mean HVAC Label')
    ax.set_title('Experiment 1: HVAC Text Variation')
    ax.set_xticks(x)
    ax.set_xticklabels(hvac_ids, rotation=45)

    plt.tight_layout()
    plt.savefig(results_dir / 'experiment1_text_variation_hvac.png')
    plt.close()

    ins_ids = [entry['text_id'] for entry in text_variation_results['insulation']]
    mean_ins_values = [entry['mean_insulation'] for entry in text_variation_results['insulation']]
    std_ins_values = [entry['std_insulation'] for entry in text_variation_results['insulation']]

    x = np.arange(len(ins_ids))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, mean_ins_values, yerr=std_ins_values, capsize=5)
    ax.set_xlabel('Insulation Text Sample ID')
    ax.set_ylabel('Mean Insulation Label')
    ax.set_title('Experiment 1: Insulation Text Variation')
    ax.set_xticks(x)
    ax.set_xticklabels(ins_ids, rotation=45)

    plt.tight_layout()
    plt.savefig(results_dir / 'experiment1_text_variation_insulation.png')
    plt.close()

    # Plot 4: Weight Optimization
    if include_weight_optimization:
        results = optimize_weights_results['results']

        # results is [((text_weight, sim_weight), diff), ...]
        text_weights = [w[0][0] for w in results]
        diffs = [w[1] for w in results]

        plt.figure()
        plt.plot(text_weights, diffs)  # default color, default style

        plt.xlabel("Text Weight (%)")
        plt.ylabel("Absolute Difference of Experimental Outcomes")
        plt.title("Text/Simulation Weight Sweep")

        plt.tight_layout()
        plt.savefig(results_dir / 'weight_optimization.png')
        plt.close()

