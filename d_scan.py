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

def simulation_step(positions, L, N, l_nuc, v, d, rng):
    order = rng.permutation(N)
    for idx_in_order, i in enumerate(order):
        action = rng.choice([-1, 0, 1])
        if action == 0:
            violating = []
            if i > 0:
                gap_left = positions[i] - (positions[i - 1] + l_nuc)
                if gap_left < d:
                    violating.append(-1)
            if i < N - 1:
                gap_right = positions[i + 1] - (positions[i] + l_nuc)
                if gap_right < d:
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

def run_simulation(pos_init, L, N, l_nuc, v, d, n_steps, rng):
    pos = pos_init.copy()
    history = [pos.copy()]
    report_every = max(1, n_steps // 10)
    for step in range(n_steps):
        pos = simulation_step(pos, L, N, l_nuc, v, d, rng)
        history.append(pos.copy())
        if (step + 1) % report_every == 0:
            print(f'  {(step + 1) / n_steps * 100:.0f}% ({step + 1}/{n_steps})')
    return np.array(history)

def compute_correlation(history, L, N, L_NUC):
    """碱基对状态相关性按间距分组 (修正版).

    对每个时间步 t，提取所有距离为 Δ 的碱基对 → 在空间维度
    计算 Pearson r → 得到 r_t(Δ)；最后对 30000 个时间步取平均。
    """
    T_analysis = len(history)
    binary = np.ones((T_analysis, L), dtype=np.int8)
    for t in range(T_analysis):
        for i in range(N):
            start = int(np.floor(history[t, i]))
            end = int(np.floor(history[t, i] + L_NUC))
            start = max(0, start)
            end = min(L, end)
            binary[t, start:end] = 0

    avg_corr = np.zeros(L)
    avg_corr[0] = 1.0        # Δ=0 自相关
    valid_frac = np.ones(L)  # 有效时间步比例(Δ=0 始终有效)

    for delta in range(1, L):
        n = L - delta
        # 切片成 (T, n) 的左右配对矩阵
        left  = binary[:, :n]       # (T, n) view
        right = binary[:, delta:]   # (T, n) view

        # 每时间步的空间统计量 (用 int 累加避免中间 float 膨胀)
        sum_left  = left.sum(axis=1).astype(np.float64)   # (T,)
        sum_right = right.sum(axis=1).astype(np.float64)  # (T,)
        sum_both  = (left * right).sum(axis=1).astype(np.float64)  # (T,)

        mean_left  = sum_left / n
        mean_right = sum_right / n
        mean_both  = sum_both / n

        # 二元变量: var = p * (1-p)
        var_left  = mean_left * (1.0 - mean_left)
        var_right = mean_right * (1.0 - mean_right)

        numer = mean_both - mean_left * mean_right
        denom = np.sqrt(var_left * var_right)

        valid = denom > 1e-12
        r_t = np.full(T_analysis, np.nan, dtype=np.float64)
        r_t[valid] = numer[valid] / denom[valid]

        avg_corr[delta] = np.nanmean(r_t)
        valid_frac[delta] = valid.mean()

    return avg_corr, valid_frac


def find_peaks_troughs(avg_corr, smooth_window=11, prominence=0.02):
    """检测相关性曲线上的波峰和波谷位置 (Δ > 0).

    参数
    ----
    smooth_window : 先对曲线做滑动平均的窗口大小 (bp)
    prominence   : 最小显著高度，低于此的峰/谷被忽略
    """
    # 轻平滑
    if len(avg_corr) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        y = np.convolve(avg_corr, kernel, mode='same')
    else:
        y = avg_corr.copy()

    peaks = []
    troughs = []

    # 从 Δ=3 开始 (避免 Δ=0 自相关峰; 跳过平滑带来的边界)
    for i in range(3, len(y) - 2):
        # 局部极大值 → 波峰
        if y[i] > y[i - 1] and y[i] > y[i - 2] and y[i] > y[i + 1] and y[i] > y[i + 2]:
            if y[i] > prominence:
                peaks.append((i, avg_corr[i]))
        # 局部极小值 → 波谷
        if y[i] < y[i - 1] and y[i] < y[i - 2] and y[i] < y[i + 1] and y[i] < y[i + 2]:
            if y[i] < -prominence:
                troughs.append((i, avg_corr[i]))

    return peaks, troughs


def print_peak_table(d_values, results):
    """打印所有 d 值的波峰波谷汇总表"""
    print('\n' + '=' * 90)
    print('波峰 / 波谷位置汇总  (Δ in bp)')
    print('=' * 90)

    # 表头
    header = f'{"d":>4s}  '
    header += f'{"Peaks (Δ bp)":<48s}  '
    header += f'{"Troughs (Δ bp)":<30s}'
    print(header)
    print('-' * 90)

    for d in d_values:
        avg_corr = results[d]
        peaks, troughs = find_peaks_troughs(avg_corr)

        peak_str = ', '.join(f'{p:4d} (r={v:+.3f})' for p, v in peaks[:8])
        trough_str = ', '.join(f'{p:4d} (r={v:+.3f})' for p, v in troughs[:8])

        if not peak_str:
            peak_str = '(none)'
        if not trough_str:
            trough_str = '(none)'

        print(f'{d:4d}  {peak_str:<48s}  {trough_str:<30s}')

    print('-' * 90)

    # 每个 d 的详细信息
    for d in d_values:
        avg_corr = results[d]
        peaks, troughs = find_peaks_troughs(avg_corr)
        equal_gap = (L - N * L_NUC) / (N - 1)

        print(f'\n--- d = {d} 详情 ---')
        print(f'  理论均等间距 = {equal_gap:.1f} bp,  核小体重复长度 = {equal_gap + L_NUC:.1f} bp')

        if peaks:
            print(f'  波峰 ({len(peaks)} 个):')
            for i, (pos, val) in enumerate(peaks):
                gap_ratio = pos / (equal_gap + L_NUC) if i == 0 and len(peaks) > 0 else (
                    (pos - peaks[i - 1][0]) / (equal_gap + L_NUC) if i > 0 else float('nan'))
                print(f'    第{i + 1}峰: Δ = {pos:5d} bp,  r = {val:+.4f}')

        if troughs:
            print(f'  波谷 ({len(troughs)} 个):')
            for i, (pos, val) in enumerate(troughs):
                print(f'    第{i + 1}谷: Δ = {pos:5d} bp,  r = {val:+.4f}')

# ---- 主循环 ----
results = {}
results_vf = {}
for d in d_values:
    print(f'\n=== d = {d} ===')
    rng_init = np.random.default_rng(42)
    pos = init_positions(L, N, L_NUC, d, rng_init)
    rng_sim = np.random.default_rng(43)
    print(f'运行模拟（{N_STEPS} 步）...')
    history = run_simulation(pos, L, N, L_NUC, v, d, N_STEPS, rng_sim)
    print('提取稳态区间并计算相关性...')
    steady = history[ANALYSIS_START:ANALYSIS_END + 1]
    avg_corr, valid_frac = compute_correlation(steady, L, N, L_NUC)
    results[d] = avg_corr
    results_vf[d] = valid_frac

# ---- 波峰波谷检测 ----
print_peak_table(d_values, results)

# ---- 有效数据比例报告 ----
print('\n=== 有效时间步比例 (valid_frac) ===')
print(f'{"d":>4s}  {"min valid_frac":>16s}  {"Δ where valid_frac drops"}')
for d in d_values:
    vf = results_vf[d]
    min_vf = vf[1:].min()
    bad_deltas = np.where(vf[1:] < 0.5)[0]
    if len(bad_deltas) > 0:
        bad_str = f'Δ < {bad_deltas[-1] + 1} bp'
    else:
        bad_str = 'none'
    print(f'{d:4d}  {min_vf:>16.4f}  {bad_str}')

# ---- 画图 ----
fig, ax = plt.subplots(figsize=(12, 6))
deltas = np.arange(L)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(d_values)))
for d, c in zip(d_values, colors):
    ax.plot(deltas, results[d], lw=0.9, color=c, label=f'd = {d}')
ax.axhline(0, color='black', ls='--', lw=0.6)
ax.axvline(L_NUC, color='crimson', ls=':', lw=1.2, label=f'Nucleosome = {L_NUC} bp')
ax.set_xlabel('Base-pair distance Δ (bp)')
ax.set_ylabel('Average Pearson r')
ax.set_title(f'Base-Pair Correlation vs Distance (v = {v}, varied d)')
ax.legend()
ax.set_xlim(0, L)

plt.tight_layout()
plt.savefig('plots/d_scan_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n图已保存到 plots/d_scan_correlation.png')
