"""Kymograph 热图: 6 行(d) × 2 列(V1/V2), 从 50000-100000 步随机抽样 10000 条."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 9

L = 2000
N = 9
L_NUC = 147
v = 5
N_STEPS = 100000
RECORD_START = 50000
RECORD_END   = 100000
N_SAMPLE = 10000                         # 随机抽样条数
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
                proposed = positions[i] - violating[0] * v
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

def positions_to_row(pos, L, N, l_nuc):
    """将核小体位置转为 0/1 行 (0=覆盖, 1=开放)."""
    row = np.ones(L, dtype=np.int8)
    for i in range(N):
        s = int(np.floor(pos[i]))
        e = int(np.floor(pos[i] + l_nuc))
        row[max(0, s):min(L, e)] = 0
    return row

def run_and_record(pos_init, L, N, l_nuc, v, d, n_steps, rng, step_func,
                   record_start, record_end):
    pos = pos_init.copy()
    T_record = record_end - record_start
    kymo = np.zeros((T_record, L), dtype=np.int8)

    report_every = max(1, n_steps // 10)
    for step in range(n_steps):
        pos = step_func(pos, L, N, l_nuc, v, d, rng)
        if record_start <= step + 1 < record_end:
            t = step + 1 - record_start
            kymo[t] = positions_to_row(pos, L, N, l_nuc)
        if (step + 1) % report_every == 0:
            print(f'    {(step + 1) / n_steps * 100:.0f}% ({step + 1}/{n_steps})')
    return kymo

# ---- 主循环 ----
kymos = {}
rng_sample = np.random.default_rng(99)  # 固定抽样种子

for version, step_func in [('v1', simulation_step_v1), ('v2', simulation_step_v2)]:
    for d in d_values:
        key = (version, d)
        print(f'\n=== {version.upper()}  d = {d} ===')
        rng_init = np.random.default_rng(42)
        pos = init_positions(L, N, L_NUC, d, rng_init)
        rng_sim = np.random.default_rng(43)
        print(f'  模拟 {N_STEPS} 步, 记录 {RECORD_START}-{RECORD_END},'
              f' 随机抽样 {N_SAMPLE} 条...')
        kymo_full = run_and_record(pos, L, N, L_NUC, v, d, N_STEPS, rng_sim,
                                   step_func, RECORD_START, RECORD_END)
        # 从 50000 条记录中有放回随机抽 10000 条
        idx = rng_sample.choice(len(kymo_full), size=N_SAMPLE, replace=True)
        kymo_sample = kymo_full[idx]
        print(f'  完成。抽样: {kymo_full.shape} → {kymo_sample.shape}')
        kymos[key] = kymo_sample

# ---- 画图: 6 行 × 2 列 ----
n_rows = len(d_values)
n_show = min(N_SAMPLE, 2000)
stride = max(1, N_SAMPLE // n_show)

fig = plt.figure(figsize=(16, 2.8 * n_rows))
gs = GridSpec(n_rows, 2, hspace=0.45, wspace=0.10)

for row, d in enumerate(d_values):
    for col, version in enumerate(['v1', 'v2']):
        ax = fig.add_subplot(gs[row, col])
        kymo = kymos[(version, d)]
        kymo_show = kymo[::stride]

        ax.imshow(kymo_show, cmap='Greys', vmin=0, vmax=1,
                  aspect='auto', interpolation='none',
                  extent=[0, L, N_SAMPLE, 0])

        if row == 0:
            title = 'V1: 2-choice (no stop)' if version == 'v1' else 'V2: 3-choice + recoil'
            ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel(f'd = {d}\n({N_SAMPLE} random samples)')
        if row == n_rows - 1:
            ax.set_xlabel('Base-pair position (bp)')
        else:
            ax.set_xlabel('')
        ax.set_yticks([])

# 图例说明
fig.suptitle(f'DNA Binary State Kymograph  (v = {v})'
             f'\nblack = nucleosome (0), white = linker (1)',
             fontsize=11, y=1.01)

plt.savefig('plots/kymograph_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n热图已保存到 plots/kymograph_heatmap.png')
