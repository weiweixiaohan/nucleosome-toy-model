"""碱基对状态相关性 2D 热图: 行 = d 值, 列 = V1/V2."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 10

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

def simulation_step_v2(positions, L, N, l_nuc, v, d, rng):
    order = rng.permutation(N)
    for idx_in_order, i in enumerate(order):
        action = rng.choice([-1, 0, 1])
        if action == 0:
            violating = []
            if i > 0 and positions[i] - (positions[i - 1] + l_nuc) < d:
                violating.append(-1)
            if i < N - 1 and positions[i + 1] - (positions[i] + l_nuc) < d:
                violating.append(+1)
            if len(violating) == 1:
                away_direction = -violating[0]
                proposed = positions[i] + away_direction * v
                proposed = max(0.0, min(proposed, L - l_nuc))
                valid = True
                if i > 0 and proposed - (positions[i - 1] + l_nuc) < d:
                    valid = False
                if i < N - 1 and positions[i + 1] - (proposed + l_nuc) < d:
                    valid = False
                if valid:
                    positions[i] = proposed
        else:
            direction = action
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

def simulation_step_v1(positions, L, N, l_nuc, v, d, rng):
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

def build_corr(history, L, N, L_NUC):
    """从历史轨迹构建相关性矩阵."""
    T = len(history)
    binary = np.ones((T, L), dtype=np.int8)
    for t in range(T):
        for i in range(N):
            start = int(np.floor(history[t, i]))
            end = int(np.floor(history[t, i] + L_NUC))
            start = max(0, start)
            end = min(L, end)
            binary[t, start:end] = 0
    return np.corrcoef(binary.T)

# ---- 主循环 ----
corr_matrices = {}  # {(version, d): corr_matrix}

for version, step_func in [('v1', simulation_step_v1), ('v2', simulation_step_v2)]:
    for d in d_values:
        key = (version, d)
        print(f'\n=== {version.upper()}  d = {d} ===')
        rng_init = np.random.default_rng(42)
        pos = init_positions(L, N, L_NUC, d, rng_init)
        rng_sim = np.random.default_rng(43)
        print(f'运行模拟（{N_STEPS} 步）...')
        history = run_simulation(pos, L, N, L_NUC, v, d, N_STEPS, rng_sim, step_func)
        print('计算相关性矩阵...')
        steady = history[ANALYSIS_START:ANALYSIS_END + 1]
        corr_matrices[key] = build_corr(steady, L, N, L_NUC)

# ---- 画图: 6 行 × 2 列 ----
n_rows = len(d_values)
n_cols = 2
fig = plt.figure(figsize=(14, 3.2 * n_rows))
gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1, 1, 0.04],
              hspace=0.35, wspace=0.12)

vmin, vmax = -1, 1

for row, d in enumerate(d_values):
    for col, version in enumerate(['v1', 'v2']):
        ax = fig.add_subplot(gs[row, col])
        corr = corr_matrices[(version, d)]
        im = ax.imshow(corr, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       extent=[0, L, L, 0], aspect='equal', rasterized=True)
        # 核小体长度辅助线
        ax.plot([0, L], [L_NUC, L + L_NUC], 'k--', lw=0.4, alpha=0.25)
        ax.plot([L_NUC, L + L_NUC], [0, L], 'k--', lw=0.4, alpha=0.25)
        if row == 0:
            title = 'V1: 2-choice (no stop)' if version == 'v1' else 'V2: 3-choice + recoil'
            ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel(f'd = {d}\nposition (bp)', fontsize=9)
        else:
            ax.set_ylabel('')
        if row == n_rows - 1:
            ax.set_xlabel('position (bp)')
        else:
            ax.set_xlabel('')

# 右侧独立 colorbar
cbar_ax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Pearson r')

fig.suptitle(f'Base-Pair State Correlation Matrix  (v = {v})', fontsize=13, y=1.01)
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n热图已保存到 plots/correlation_heatmap.png')
