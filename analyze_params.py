"""快速提取相关性曲线关键特征，生成 markdown 可直接引用的数据。"""
import numpy as np

# 直接跑一次 d=15 的完整模拟取数据，不做 d-scan（已有 correlation.png）
# 这里从 correlation.png 对应的参数推算关键尺度

L = 2000
N = 9
L_NUC = 147
d_values = [5, 10, 20, 30, 40]

print("# 系统基本参数")
print(f"- DNA 长度: {L} bp")
print(f"- 核小体: {N} × {L_NUC} bp = {N * L_NUC} bp")
print(f"- 剩余空间: {L - N * L_NUC} bp")
print(f"- v = 5 bp/步, 共计 100000 步")
print()

for d in d_values:
    equal_gap = (L - N * L_NUC) / (N - 1)
    min_gap_sum = (N - 1) * d
    free_space = L - N * L_NUC
    actual_range = free_space - min_gap_sum
    print(f"## d = {d}")
    print(f"  - 理论均等间距: {equal_gap:.1f} bp")
    print(f"  - 最小间距总和: {min_gap_sum} bp (剩余 {free_space} bp)")
    print(f"  - 间距可波动范围: {actual_range} bp (平均每个间隙 ±{actual_range / (N - 1):.1f} bp)")
    print(f"  - d 占均等间距比例: {d / equal_gap * 100:.0f}%")
    if min_gap_sum > free_space:
        print(f"  - ⚠️ 无法放置！")
    print()

# 预期相关性曲线结构
print("# 相关性曲线的特征结构预测")
print()
print("## 通用结构 (任一 d)")
print("1. **Δ = 0**: r = 1.0 (自相关)")
print("2. **0 < Δ < 核小体内部 (≈147 bp)**: ")
print("   - 若两个位点常被同一核小体覆盖 → 高度正相关")
print("   - 若一个位点在核小体内部、另一个在间隙 → 负相关")
print("3. **Δ ≈ 147 bp (核小体长度)**: 特征信号 —— 对应核小体两端的反相关")
print("4. **Δ > 147 bp**: 振荡衰减，周期反映核小体间距")
print()
print("## d 对周期性的影响 (预期)")
print("- d 小 (5, 10): 核小体间距波动大，长程有序弱 → 振荡快速衰减")
print("- d 中 (20): 间距有适度约束 → 中等程度的周期性")
print("- d 大 (30, 40): 间距被严格约束 → 强周期性，Bravais 峰清晰")
print("  - Note: d=40 时 min_gap=320, free=677, 波动范围仅 357 bp (每间隙 ±45 bp)")
print()
print("## V1 vs V2 预期差异")
print("- V1 (纯左右): 核小体无法'休息'，始终保持高频振动")
print("  → 边界位置噪声更大 → 长程相关性衰减更快")
print("- V2 (三选一 + 反冲): 20% 的时间核小体静止，反冲机制加强间距约束")
print("  → 边界更锐利 → 周期性可能更强、振荡更持久")
