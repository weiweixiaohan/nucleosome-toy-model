import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 11

L = 2000
N = 9
L_NUC = 147
v = 5
N_STEPS = 100000
ANALYSIS_START = 60000
ANALYSIS_END = 90000
d_values = [5, 10, 20, 30, 40, 50]

def init_positions(L, N, l_nuc, d, rng):
    positions = np.zeros(N)
    for i in range(N):
        if i == 0:
            min_pos = 0.0
        else:
            min_pos = positions[i - 1] + l_nuc + d
        remaining = N - i - 1
        max_pos = L - l_nuc - remaining * (l_nuc + d)
        positions[i] = rng.uniform(min_pos, max_pos)
    return positions

def simulation_step_v1(positions, L, N, l_nuc, v, d, rng):
    """V1: 只有左右运动（各 1/2），无停顿，无反冲"""
    order = rng.permutation(N)
    for idx_in_order, i in enumerate(order):
        direction = rng.choice([-1, 1])
        for attempt in range(2):
            proposed = positions[i] + direction * v
            proposed = max(0.0, min(proposed, L - l_nuc))
            valid = True
            if i > 0 and proposed - (positions[i - 1] + l_nuc) < d:
                valid = False
            if i < N - 1 and positions[i + 1] - (proposed + l_nuc) < d:
                valid = False
            if valid:
                positions[i] = proposed
                break
            else:
                direction = -direction
    return positions

def run_simulation(pos_init, L, N, l_nuc, v, d, n_steps, rng, step_func):
    pos = pos_init.copy()
    history = [pos.copy()]
    report_every = max(1, n_steps // 10)
    for step in range(n_steps):
        pos = step_func(pos, L, N, l_nuc, v, d, rng)
        history.append(pos.copy())
        if (step + 1) % report_every == 0:
            print(f'  {(step + 1) / n_steps * 100:.0f}% ({step + 1}/{n_steps})')
    return np.array(history)

def compute_correlation(history, L, N, L_NUC):
    T_analysis = len(history)
    binary = np.ones((T_analysis, L), dtype=np.int8)
    for t in range(T_analysis):
        for i in range(N):
            start = int(np.floor(history[t, i]))
            end = int(np.floor(history[t, i] + L_NUC))
            start = max(0, start)
            end = min(L, end)
            binary[t, start:end] = 0
    corr = np.corrcoef(binary.T)
    max_delta = L - 1
    avg_corr = np.zeros(max_delta + 1)
    for delta in range(max_delta + 1):
        avg_corr[delta] = np.diag(corr, k=delta).mean()
    return avg_corr

# ---- 主循环 ----
results_v1 = {}
for d in d_values:
    print(f'\n=== V1 (2-choice) d = {d} ===')
    rng_init = np.random.default_rng(42)
    pos = init_positions(L, N, L_NUC, d, rng_init)
    rng_sim = np.random.default_rng(43)
    print(f'运行模拟（{N_STEPS} 步）...')
    history = run_simulation(pos, L, N, L_NUC, v, d, N_STEPS, rng_sim, simulation_step_v1)
    print('提取稳态区间并计算相关性...')
    steady = history[ANALYSIS_START:ANALYSIS_END + 1]
    avg_corr = compute_correlation(steady, L, N, L_NUC)
    results_v1[d] = avg_corr

# ---- 画图 ----
fig, ax = plt.subplots(figsize=(12, 6))
deltas = np.arange(L)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(d_values)))
for d, c in zip(d_values, colors):
    ax.plot(deltas, results_v1[d], lw=0.9, color=c, label=f'd = {d}')
ax.axhline(0, color='black', ls='--', lw=0.6)
ax.axvline(L_NUC, color='crimson', ls=':', lw=1.2, label=f'Nucleosome = {L_NUC} bp')
ax.set_xlabel('Base-pair distance Δ (bp)')
ax.set_ylabel('Average Pearson r')
ax.set_title(f'V1 (2-choice, no stop/recoil): Correlation vs Distance (v = {v})')
ax.legend()
ax.set_xlim(0, L)

plt.tight_layout()
plt.savefig('plots/d_scan_v1_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n图已保存到 plots/d_scan_v1_correlation.png')
