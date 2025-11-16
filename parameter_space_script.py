import subprocess
import sys
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def run_advanced_root_finder(a, b, n, sims, master_pickle_file):
    command = [
        sys.executable,
        'ADVANCED_ROOT_FINDING_SCRIPT.py',
        '--a', str(a),
        '--b', str(b),
        '--n', str(n),
        '--sims', str(sims),
        '--master-pickle', master_pickle_file
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )

    try:
        parsed = json.loads(result.stdout)
        return parsed
    except json.JSONDecodeError:
        print("Could not parse JSON output:")
        print(result.stdout)
        return None


if __name__ == "__main__":

    n = 2
    sims = 100
    a_lower = 0.0001; a_upper = 100
    b_lower = 500; b_upper = 1e8
    a_size = 3
    b_size = 3
    
    # Create master pickle filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_pickle_file = f"master_multistability_n{n}_sims{sims}_grid{a_size}x{b_size}_{timestamp}.pkl"
    
    # Delete master file if it exists (fresh start)
    if os.path.exists(master_pickle_file):
        os.remove(master_pickle_file)
    
    print(f"Master pickle file: {master_pickle_file}")
    print(f"Total expected entries: {a_size * b_size * sims}")
    
    a_array = np.linspace(np.log10(a_lower), np.log10(a_upper), a_size)
    b_array = np.linspace(np.log10(b_lower), np.log10(b_upper), b_size)

    a_array = 10**a_array
    b_array = 10**b_array

    num_multistability_grid = np.full((a_size, b_size), np.nan, dtype=float)
    mean_multistability_grid = np.full((a_size, b_size), np.nan, dtype=float)
    total = a_size * b_size
    idx = 0

    for j, b in enumerate(b_array):          
        for i, a in enumerate(a_array):      
            idx += 1
            print(f"[{idx}/{total}] a={a:.6g}, b={b:.6g} ...", end=" ")
            res = run_advanced_root_finder(a, b, n, sims, master_pickle_file)
            if res is None:
                print("no result")
                continue
            num_ms = res.get("num_multistable_systems") if isinstance(res, dict) else None
            mean_ms = res.get("average_num_stable_states") if isinstance(res, dict) else None
            total_in_master = res.get("total_results_in_master", 0)

            num_truly_multistable = res.get("num_truly_multistable", 0)

            if int(num_truly_multistable) == 0:
                mean_ms = 0
            pct = 100 * float(num_truly_multistable) / float(sims)
            num_multistability_grid[i, j] = pct
            mean_multistability_grid[i, j] = mean_ms
            print(f"{num_ms}/{sims} -> {pct:.2f}% ... mean # ms = {mean_ms} ... master total: {total_in_master}")

    # print(f"\n=== Grid scan complete ===")
    # print(f"Master pickle file: {master_pickle_file}")
    # print(f"Expected total entries: {a_size * b_size * sims}")
    
    a_edges = np.logspace(np.log10(a_lower), np.log10(a_upper), a_size + 1)
    b_edges = np.logspace(np.log10(b_lower), np.log10(b_upper), b_size + 1)
    A_mesh, B_mesh = np.meshgrid(a_edges, b_edges)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: percent multistable
    im0 = axes[0].pcolormesh(A_mesh, B_mesh, num_multistability_grid.T, shading='auto', cmap='inferno')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(a_lower, a_upper)
    axes[0].set_ylim(b_lower, b_upper)
    axes[0].set_xlabel(r"$a$")
    axes[0].set_ylabel(r"$b$")
    axes[0].set_title("% multistable outcomes")
    cbar0 = fig.colorbar(im0, ax=axes[0], label="% multistable outcomes")

    # Right: mean number of stable states
    im1 = axes[1].pcolormesh(A_mesh, B_mesh, mean_multistability_grid.T, shading='auto', cmap='viridis')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(a_lower, a_upper)
    axes[1].set_ylim(b_lower, b_upper)
    axes[1].set_xlabel(r"$a$")
    axes[1].set_title("mean number of stable states")
    cbar1 = fig.colorbar(im1, ax=axes[1], label="mean stable states")

    # fig.suptitle(f"sims per point = {sims}\nMaster file: {master_pickle_file}", y=1.02)
    
    # Save with timestamp
    # plot_file = f"parameter_space_2panel_{timestamp}.png"
    # plt.savefig(plot_file, dpi=400, bbox_inches='tight')
    # print(f"Plot saved to: {plot_file}")
    plt.show()