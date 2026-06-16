"""
综合测试案例 - 验证所有算法在Müller-Brown势能面上的表现

案例1: 牛顿法从不同初始点搜索极小值和鞍点
案例2: Dimer方法完整过程
案例3: NEB vs CI-NEB 对比
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs'))

import numpy as np
from pes import MuellerBrownPES
from algorithms import NewtonMethod, DimerMethod, NEBMethod

pes = MuellerBrownPES()

print("=" * 60)
print("案例1: 牛顿法从不同初始点搜索")
print("=" * 60)

# 已知关键点
known_min1 = (-0.558, 1.442)   # 全局极小 E≈-146.7
known_min2 = (0.623, 0.028)    # 局部极小 E≈-108.2
known_saddle = (-0.822, 0.624) # 鞍点 E≈-40.7

# 从不同初始点搜索极小值
init_points_min = [
    (np.array([-0.5, 1.5]), "靠近全局极小"),
    (np.array([0.5, 0.5]), "靠近局部极小"),
    (np.array([-1.0, 1.0]), "中间区域"),
]

print("\n--- 牛顿法(极小值) ---")
for start, desc in init_points_min:
    nm = NewtonMethod(pes, start, hessian_sign='positive')
    traj = nm.run()
    pos = nm.position
    E = pes.energy(pos[0], pos[1])
    print(f"  起点{start} ({desc}):")
    print(f"    收敛={nm.is_converged}, 终点=({pos[0]:.4f},{pos[1]:.4f}), E={E:.4f}, 步数={len(traj)}")

# 从不同初始点搜索鞍点
init_points_saddle = [
    (np.array([-0.8, 0.65]), "靠近鞍点"),
    (np.array([-0.7, 0.7]), "鞍点附近"),
]

print("\n--- 牛顿法(鞍点) ---")
for start, desc in init_points_saddle:
    nm = NewtonMethod(pes, start, hessian_sign='negative')
    traj = nm.run()
    pos = nm.position
    E = pes.energy(pos[0], pos[1])
    print(f"  起点{start} ({desc}):")
    print(f"    收敛={nm.is_converged}, 终点=({pos[0]:.4f},{pos[1]:.4f}), E={E:.4f}, 步数={len(traj)}")

print("\n" + "=" * 60)
print("案例2: Dimer方法完整过程")
print("=" * 60)

dimer_starts = [
    (np.array([-0.8, 0.65]), np.array([-0.4, -0.6]), "鞍点附近"),
    (np.array([-0.5, 1.0]), np.array([0.5, -0.5]), "较远处"),
]

for start, direction, desc in dimer_starts:
    dm = DimerMethod(pes, start, start_direction=direction)
    traj = dm.run()
    pos = dm.center
    E = pes.energy(pos[0], pos[1])
    print(f"\n  起点{start}, 方向{direction} ({desc}):")
    print(f"    收敛={dm.is_converged}, 终点=({pos[0]:.4f},{pos[1]:.4f}), E={E:.4f}, 步数={len(traj)}")

    # 显示前几步的详细信息
    for i, s in enumerate(traj[:5]):
        c = s['center']
        print(f"    Step {i}: pos=({c[0]:.4f},{c[1]:.4f}), E={s['energy']:.4f}, "
              f"C={s['curvature']:.2f}, |F|={s['gradient_norm']:.4f}")

print("\n" + "=" * 60)
print("案例3: NEB vs CI-NEB 对比")
print("=" * 60)

min1 = np.array([-0.558, 1.442])
min2 = np.array([0.623, 0.028])

# NEB
print("\n--- NEB方法 ---")
neb = NEBMethod(pes, min1, min2, n_images=15, spring_k=5.0,
                conv_threshold=1.0, max_iter=500, climbing_image=False)
neb_traj = neb.run()
neb_energies = np.array([pes.energy(p[0], p[1]) for p in neb.images])
neb_barrier = np.max(neb_energies) - np.min(neb_energies)
print(f"  收敛={neb.is_converged}, 步数={len(neb_traj)}")
print(f"  能量范围: {np.min(neb_energies):.4f} ~ {np.max(neb_energies):.4f}")
print(f"  势垒高度: {neb_barrier:.4f}")

# CI-NEB
print("\n--- CI-NEB方法 ---")
ci_neb = NEBMethod(pes, min1, min2, n_images=15, spring_k=5.0,
                   conv_threshold=1.0, max_iter=500, climbing_image=True)
ci_traj = ci_neb.run()
ci_energies = np.array([pes.energy(p[0], p[1]) for p in ci_neb.images])
ci_barrier = np.max(ci_energies) - np.min(ci_energies)
ci_idx = int(np.argmax(ci_energies[1:-1]) + 1)
ci_pos = ci_neb.images[ci_idx]
print(f"  收敛={ci_neb.is_converged}, 步数={len(ci_traj)}")
print(f"  能量范围: {np.min(ci_energies):.4f} ~ {np.max(ci_energies):.4f}")
print(f"  势垒高度: {ci_barrier:.4f}")
print(f"  CI点位置: ({ci_pos[0]:.4f}, {ci_pos[1]:.4f}), E={ci_energies[ci_idx]:.4f}")

# 对比
print("\n--- 对比结果 ---")
print(f"  {'方法':<10} {'收敛':<6} {'步数':<6} {'势垒高度':<12} {'最高能量':<12}")
print(f"  {'NEB':<10} {str(neb.is_converged):<6} {len(neb_traj):<6} {neb_barrier:<12.4f} {np.max(neb_energies):<12.4f}")
print(f"  {'CI-NEB':<10} {str(ci_neb.is_converged):<6} {len(ci_traj):<6} {ci_barrier:<12.4f} {np.max(ci_energies):<12.4f}")
print(f"\n  已知鞍点: ({known_saddle[0]}, {known_saddle[1]}), E={pes.energy(known_saddle[0], known_saddle[1]):.4f}")

print("\n" + "=" * 60)
print("所有测试案例完成！")
print("=" * 60)
