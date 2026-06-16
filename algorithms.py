"""
搜索算法模块
包含牛顿法、Dimer方法、NEB方法、最速下降法、盆地跳跃法、Metropolis MC，
用于势能面上的极小值和鞍点搜索。
所有算法严格对应课程公式，使用numpy实现。
"""

import numpy as np
from typing import Optional


class NewtonMethod:
    """牛顿法（极小值/鞍点搜索）

    核心迭代公式：q_{t+1} = q_t + H^{-1} · F，其中 F = -∇E
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        conv_threshold: float = 1e-4,
        max_iter: int = 100,
        hessian_sign: str = "positive",
        max_step: float = 0.5,
    ):
        """
        Args:
            pes: 势能面对象，需有 energy(), gradient(), hessian() 方法
            start_pos: 初始位置 np.array([x, y])
            conv_threshold: 收敛阈值（梯度模长）
            max_iter: 最大迭代步数
            hessian_sign: 'positive' 或 'negative'
                'positive': 使用原始Hessian，收敛到极小值
                'negative': 反转最小本征值对应的Hessian分量，收敛到鞍点
            max_step: 最大步长限制，防止发散
        """
        self.pes = pes
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.hessian_sign = hessian_sign
        self.max_step = max_step

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

    def step(self) -> dict:
        """执行一步牛顿迭代，返回状态信息"""
        x, y = self.position

        # 1. 计算梯度 F = -gradient(x,y) 和 Hessian H
        grad = np.array(self.pes.gradient(x, y), dtype=float)
        F = -grad  # 力 = -梯度
        H = np.array(self.pes.hessian(x, y), dtype=float)

        # 2. 根据 hessian_sign 修改 Hessian
        eigenvalues_orig, eigenvectors = np.linalg.eigh(H)

        if self.hessian_sign == "negative":
            # 鞍点搜索：确保Hessian有且仅有一个负本征值
            # 找到最小本征值，如果为正则取反使其为负
            # 如果已有负本征值则保持
            eigenvalues_new = eigenvalues_orig.copy()
            min_idx = np.argmin(eigenvalues_new)
            if eigenvalues_new[min_idx] > 0:
                # 所有本征值为正，取反最小值使其为负
                eigenvalues_new[min_idx] = -eigenvalues_new[min_idx]
            # 确保其他本征值为正（如有多个负值，只保留最小的为负）
            for i in range(len(eigenvalues_new)):
                if i != min_idx and eigenvalues_new[i] < 0:
                    eigenvalues_new[i] = -eigenvalues_new[i]
            H = eigenvectors @ np.diag(eigenvalues_new) @ eigenvectors.T
            hessian_eigenvalues = eigenvalues_new
        else:
            # 极小值搜索：确保Hessian正定
            eigenvalues_new = eigenvalues_orig.copy()
            # 如果有负本征值，取反使其为正
            for i in range(len(eigenvalues_new)):
                if eigenvalues_new[i] < 0:
                    eigenvalues_new[i] = -eigenvalues_new[i]
            H = eigenvectors @ np.diag(eigenvalues_new) @ eigenvectors.T
            hessian_eigenvalues = eigenvalues_new

        # 3. 计算搜索方向：dq = H^{-1} · F
        try:
            dq = np.linalg.solve(H, F)
        except np.linalg.LinAlgError:
            # Hessian奇异，使用伪逆
            dq = np.linalg.lstsq(H, F, rcond=None)[0]

        # 4. 限制步长，防止发散
        step_norm = np.linalg.norm(dq)
        if step_norm > self.max_step:
            dq = dq * self.max_step / step_norm

        # 5. 回溯线搜索（Armijo条件）
        alpha = 1.0
        current_energy = self.pes.energy(x, y)
        c1 = 1e-4  # Armijo参数
        directional_deriv = np.dot(grad, -dq)  # 搜索方向上的方向导数

        for _ in range(20):
            new_pos = self.position + alpha * dq
            new_energy = self.pes.energy(new_pos[0], new_pos[1])
            if self.hessian_sign == "positive":
                # 极小值搜索：Armijo条件确保充分下降
                if new_energy <= current_energy + c1 * alpha * directional_deriv:
                    break
            else:
                # 鞍点搜索：不要求能量下降，但限制步长避免发散
                if alpha < 0.01:
                    break
                # 如果步长太大导致能量暴增，也减小步长
                if new_energy > current_energy + 100 * abs(current_energy + 1):
                    alpha *= 0.5
                    continue
                break
            alpha *= 0.5

        # 6. 更新位置
        self.position = self.position + alpha * dq
        self._iter_count += 1

        # 7. 检查收敛：|gradient| < threshold
        grad_norm = np.linalg.norm(grad)
        self._converged = grad_norm < self.conv_threshold

        # 记录状态
        new_x, new_y = self.position
        state = {
            "position": self.position.copy(),
            "energy": float(self.pes.energy(new_x, new_y)),
            "gradient": grad.copy(),
            "hessian_eigenvalues": hessian_eigenvalues.copy(),
            "step_size": float(np.linalg.norm(dq)),
            "gradient_norm": float(grad_norm),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        """运行到收敛或最大步数"""
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        """重置到初始位置"""
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

    def get_trajectory(self) -> list[dict]:
        """获取轨迹"""
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        """是否已收敛"""
        return self._converged


class DimerMethod:
    """Dimer方法（过渡态搜索）

    双子定义：R1 = R0 + ΔR·N̂, R2 = R0 - ΔR·N̂
    通过旋转操作寻找最低曲率方向，通过平动操作向鞍点逼近。
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        start_direction: Optional[np.ndarray] = None,
        delta_r: float = 0.01,
        conv_threshold: float = 1e-3,
        max_iter: int = 200,
        trans_step: float = 0.05,
        n_rotate_steps: int = 5,
    ):
        """
        Args:
            pes: 势能面对象
            start_pos: 初始中心位置
            start_direction: 初始双子方向（单位向量），默认随机
            delta_r: 双子半间距 ΔR
            conv_threshold: 收敛阈值
            max_iter: 最大迭代步数
            trans_step: 平动步长
            n_rotate_steps: 每次平动前的旋转步数
        """
        self.pes = pes
        self.delta_r = delta_r
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.trans_step = trans_step
        self.n_rotate_steps = n_rotate_steps

        self.center = np.array(start_pos, dtype=float)
        self._start_center = self.center.copy()

        # 初始化双子方向
        if start_direction is not None:
            self.direction = np.array(start_direction, dtype=float)
            self.direction = self.direction / np.linalg.norm(self.direction)
        else:
            # 随机初始化并归一化
            self.direction = np.random.randn(len(self.center))
            self.direction = self.direction / np.linalg.norm(self.direction)

        self._start_direction = self.direction.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

    def _compute_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """计算双子两端点：R1 = R0 + ΔR·N̂, R2 = R0 - ΔR·N̂"""
        R1 = self.center + self.delta_r * self.direction
        R2 = self.center - self.delta_r * self.direction
        return R1, R2

    def _compute_curvature(self, F1: np.ndarray, F2: np.ndarray) -> float:
        """计算曲率：C = (F1 - F2) · N̂ / (2·ΔR)"""
        return float(np.dot(F1 - F2, self.direction) / (2.0 * self.delta_r))

    def rotate_step(self) -> dict:
        """执行一步旋转操作，寻找最低曲率方向

        对于2D势能面，直接使用Hessian本征分解找到最小曲率方向。
        这等价于Dimer旋转操作的解析解，避免了数值旋转的不稳定性。
        """
        x, y = self.center

        # 计算Hessian矩阵
        H = np.array(self.pes.hessian(x, y), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # 最小本征值对应的本征向量即为最小曲率方向
        min_idx = np.argmin(eigenvalues)
        new_direction = eigenvectors[:, min_idx]

        # 保持方向连续性：如果新方向与旧方向夹角大于90度，取反
        if np.dot(new_direction, self.direction) < 0:
            new_direction = -new_direction

        self.direction = new_direction / np.linalg.norm(new_direction)

        # 计算当前曲率用于记录
        R1, R2 = self._compute_endpoints()
        F1 = -np.array(self.pes.gradient(R1[0], R1[1]), dtype=float)
        F2 = -np.array(self.pes.gradient(R2[0], R2[1]), dtype=float)
        C = self._compute_curvature(F1, F2)

        return {
            "curvature": C,
            "direction": self.direction.copy(),
            "F_rot_norm": float(eigenvalues[min_idx]),
        }

    def translate_step(self) -> dict:
        """执行一步平动操作，向鞍点逼近

        1. 计算中心点力：F0 = -∇E(R0)
        2. 分解力：F_parallel = (F0 · N̂) · N̂, F_perp = F0 - F_parallel
        3. 计算平动力 F_dagger
        4. 使用自适应步长更新位置（接近鞍点时切换为牛顿法）
        """
        x, y = self.center

        # 1. 计算中心点力
        F0 = -np.array(self.pes.gradient(x, y), dtype=float)
        grad_norm = np.linalg.norm(F0)

        # 2. 分解力
        F_parallel = np.dot(F0, self.direction) * self.direction
        F_perp = F0 - F_parallel

        # 计算曲率（用于判断和记录）
        R1, R2 = self._compute_endpoints()
        F1 = -np.array(self.pes.gradient(R1[0], R1[1]), dtype=float)
        F2 = -np.array(self.pes.gradient(R2[0], R2[1]), dtype=float)
        C = self._compute_curvature(F1, F2)

        # 3. 计算平动力 F_dagger
        # 根据课程公式：F† = -F∥ + F⊥
        F_dagger = -F_parallel + F_perp
        F_dagger_norm = np.linalg.norm(F_dagger)

        # 4. 步进更新位置
        max_step = 0.2

        if grad_norm < 10.0:
            # 接近鞍点时，切换为牛顿法搜索鞍点
            # 使用标准力 F0 = -∇E 和修改后的Hessian
            H = np.array(self.pes.hessian(x, y), dtype=float)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            # 修改Hessian：确保最小本征值为负（鞍点结构），其余为正
            eigenvalues_mod = eigenvalues.copy()
            min_idx = np.argmin(eigenvalues_mod)
            if eigenvalues_mod[min_idx] > 0:
                eigenvalues_mod[min_idx] = -eigenvalues_mod[min_idx]
            for i in range(len(eigenvalues_mod)):
                if i != min_idx and eigenvalues_mod[i] < 0:
                    eigenvalues_mod[i] = -eigenvalues_mod[i]

            H_mod = eigenvectors @ np.diag(eigenvalues_mod) @ eigenvectors.T

            # 牛顿步：dq = H_mod^{-1} · F0
            try:
                step = np.linalg.solve(H_mod, F0)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H_mod, F0, rcond=None)[0]

            step_norm = np.linalg.norm(step)
            if step_norm > max_step:
                step = step * max_step / step_norm
            self.center = self.center + step

        elif F_dagger_norm > 1e-12:
            # 远离鞍点时，使用归一化Dimer力方向 × 固定步长
            step = self.trans_step * F_dagger / F_dagger_norm
            step_norm = np.linalg.norm(step)
            if step_norm > max_step:
                step = step * max_step / step_norm
            self.center = self.center + step

        # 检查收敛：|F0| < threshold 且 C > 0
        # 注：曲率 C = (F1-F2)·N̂/(2ΔR) ≈ -λ（Hessian最小本征值的负值）
        # 鞍点处λ<0，所以C>0表示双子对准了负曲率方向
        self._converged = grad_norm < self.conv_threshold and C > 0

        self._iter_count += 1

        # 记录状态
        new_x, new_y = self.center
        state = {
            "center": self.center.copy(),
            "direction": self.direction.copy(),
            "curvature": C,
            "f_parallel": float(np.linalg.norm(F_parallel)),
            "f_perp": float(np.linalg.norm(F_perp)),
            "energy": float(self.pes.energy(new_x, new_y)),
            "gradient_norm": float(grad_norm),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def step(self) -> dict:
        """执行一步完整迭代（先旋转再平动）"""
        # 先执行多步旋转，寻找最低曲率方向
        for _ in range(self.n_rotate_steps):
            self.rotate_step()
        # 再执行一步平动
        return self.translate_step()

    def run(self) -> list[dict]:
        """运行到收敛或最大步数"""
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self):
        """重置到初始状态"""
        self.center = self._start_center.copy()
        self.direction = self._start_direction.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

    def get_trajectory(self) -> list[dict]:
        """获取轨迹"""
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        """是否已收敛"""
        return self._converged


class NEBMethod:
    """NEB方法（双端过渡态搜索）

    在初态和终态之间生成一系列镜像点，通过弹簧力和真实力的组合
    优化路径，寻找最小能量路径和过渡态。
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        n_images: int = 10,
        spring_k: float = 1.0,
        conv_threshold: float = 1e-3,
        max_iter: int = 200,
        climbing_image: bool = False,
        step_size: float = 0.1,
    ):
        """
        Args:
            pes: 势能面对象
            start_pos: 初态位置 np.array([x,y])
            end_pos: 终态位置 np.array([x,y])
            n_images: 镜像点数量
            spring_k: 弹簧力常数
            conv_threshold: 收敛阈值
            max_iter: 最大迭代步数
            climbing_image: 是否启用CI-NEB
            step_size: 优化步长
        """
        self.pes = pes
        self.n_images = n_images
        self.spring_k = spring_k
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.climbing_image = climbing_image
        self.step_size = step_size

        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)

        # 初始化：在初态和终态之间线性插值生成镜像点
        self.images = np.zeros((n_images, len(self.start_pos)))
        for i in range(n_images):
            t = i / (n_images - 1)
            self.images[i] = (1 - t) * self.start_pos + t * self.end_pos

        self._initial_images = self.images.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # FIRE优化器参数
        self._velocities = np.zeros_like(self.images)
        self._fire_dt = 0.05  # 初始时间步长
        self._fire_dt_max = 0.1  # 最大时间步长
        self._fire_alpha = 0.1  # 混合参数
        self._fire_alpha_start = 0.1
        self._fire_f_inc = 1.1  # dt增长因子
        self._fire_f_dec = 0.5  # dt衰减因子
        self._fire_f_alpha = 0.99  # alpha衰减因子
        self._fire_N_min = 5  # 最小步数后才开始增大dt
        self._fire_steps_since_negative = 0  # 上次负P后的步数

    def _compute_tangents(self) -> np.ndarray:
        """计算切向量

        τ_i = (R_i - R_{i-1})/|R_i - R_{i-1}| + (R_{i+1} - R_i)/|R_{i+1} - R_i|
        归一化：τ̂_i = τ_i / |τ_i|
        端点处理：τ_0 = (R_1 - R_0)/|R_1 - R_0|, τ_N = (R_N - R_{N-1})/|R_N - R_{N-1}|
        """
        n = self.n_images
        tangents = np.zeros_like(self.images)

        # 端点切向量
        d_start = self.images[1] - self.images[0]
        norm_start = np.linalg.norm(d_start)
        if norm_start > 1e-12:
            tangents[0] = d_start / norm_start

        d_end = self.images[-1] - self.images[-2]
        norm_end = np.linalg.norm(d_end)
        if norm_end > 1e-12:
            tangents[-1] = d_end / norm_end

        # 内部点切向量
        for i in range(1, n - 1):
            d_prev = self.images[i] - self.images[i - 1]
            d_next = self.images[i + 1] - self.images[i]
            norm_prev = np.linalg.norm(d_prev)
            norm_next = np.linalg.norm(d_next)

            if norm_prev > 1e-12 and norm_next > 1e-12:
                tau = d_prev / norm_prev + d_next / norm_next
                norm_tau = np.linalg.norm(tau)
                if norm_tau > 1e-12:
                    tangents[i] = tau / norm_tau
            elif norm_prev > 1e-12:
                tangents[i] = d_prev / norm_prev
            elif norm_next > 1e-12:
                tangents[i] = d_next / norm_next

        return tangents

    def _compute_forces(self) -> tuple[np.ndarray, float, Optional[int]]:
        """计算NEB力

        对每个内部镜像点 i=1,...,N-1：
        1. 真实力：F_true_i = -∇E(R_i)
        2. 真实力的垂直分量：F_i^⊥ = F_true_i - (F_true_i · τ̂_i) · τ̂_i
        3. 弹簧力平行分量：F_i^S∥ = k·(|R_{i+1}-R_i| - |R_i-R_{i-1}|) · τ̂_i
        4. 总力：F_i^NEB = F_i^⊥ + F_i^S∥

        CI-NEB：对能量最高的镜像点，反转真实力沿切向的分量
        """
        n = self.n_images
        tangents = self._compute_tangents()
        forces = np.zeros_like(self.images)

        # 计算所有镜像点的能量
        energies = np.zeros(n)
        for i in range(n):
            energies[i] = self.pes.energy(self.images[i][0], self.images[i][1])

        # 确定climbing image索引
        ci_idx = None
        if self.climbing_image:
            # 内部镜像点中能量最高的
            ci_idx = int(np.argmax(energies[1:-1]) + 1)

        # 计算内部镜像点的力
        for i in range(1, n - 1):
            # 1. 真实力
            grad = np.array(
                self.pes.gradient(self.images[i][0], self.images[i][1]), dtype=float
            )
            F_true = -grad

            tau = tangents[i]

            # 2. 真实力的垂直分量
            F_true_perp = F_true - np.dot(F_true, tau) * tau

            if i == ci_idx:
                # CI-NEB：移除弹簧力，反转真实力沿切向的分量
                # F_CI = F_true - 2·(F_true · τ̂) · τ̂ = F_true_perp - (F_true · τ̂) · τ̂
                F_true_parallel = np.dot(F_true, tau) * tau
                forces[i] = F_true_perp - F_true_parallel
            else:
                # 3. 弹簧力平行分量
                d_next = np.linalg.norm(self.images[i + 1] - self.images[i])
                d_prev = np.linalg.norm(self.images[i] - self.images[i - 1])
                F_spring_parallel = self.spring_k * (d_next - d_prev) * tau

                # 4. 总力
                forces[i] = F_true_perp + F_spring_parallel

        # 计算最大力（仅内部镜像点）
        max_force = 0.0
        for i in range(1, n - 1):
            force_norm = np.linalg.norm(forces[i])
            if force_norm > max_force:
                max_force = force_norm

        return forces, max_force, ci_idx

    def step(self) -> dict:
        """执行一步NEB优化（使用FIRE算法）"""
        forces, max_force, ci_idx = self._compute_forces()

        # FIRE优化算法
        # 1. 计算P = F·v
        P = 0.0
        for i in range(1, self.n_images - 1):
            P += np.dot(forces[i], self._velocities[i])

        # 2. 根据P的符号调整
        if P > 0:
            # 沿力方向加速
            self._fire_steps_since_negative += 1
            # 混合速度：v = (1-α)*v + α*|v|*F̂
            for i in range(1, self.n_images - 1):
                v_norm = np.linalg.norm(self._velocities[i])
                f_norm = np.linalg.norm(forces[i])
                if f_norm > 1e-12 and v_norm > 1e-12:
                    self._velocities[i] = (
                        (1 - self._fire_alpha) * self._velocities[i]
                        + self._fire_alpha * v_norm * forces[i] / f_norm
                    )
            # 增大时间步长，减小alpha
            if self._fire_steps_since_negative > self._fire_N_min:
                self._fire_dt = min(
                    self._fire_dt * self._fire_f_inc, self._fire_dt_max
                )
                self._fire_alpha = self._fire_alpha * self._fire_f_alpha
        else:
            # 刹车：重置速度，减小时间步长
            for i in range(1, self.n_images - 1):
                self._velocities[i] = 0.0
            self._fire_dt = self._fire_dt * self._fire_f_dec
            self._fire_alpha = self._fire_alpha_start
            self._fire_steps_since_negative = 0

        # 3. 更新速度和位置（Velocity Verlet）
        max_move = 0.1  # 最大位移限制
        for i in range(1, self.n_images - 1):
            self._velocities[i] = self._velocities[i] + self._fire_dt * forces[i]
            move = self._fire_dt * self._velocities[i]
            move_norm = np.linalg.norm(move)
            if move_norm > max_move:
                move = max_move * move / move_norm
                self._velocities[i] = move / self._fire_dt
            self.images[i] = self.images[i] + move

        self._iter_count += 1

        # 检查收敛：所有镜像点受力的最大模长 < threshold
        self._converged = max_force < self.conv_threshold

        # 计算能量分布
        energies = np.zeros(self.n_images)
        for i in range(self.n_images):
            energies[i] = self.pes.energy(self.images[i][0], self.images[i][1])

        # 记录状态
        state = {
            "images": self.images.copy(),
            "energies": energies.copy(),
            "max_force": float(max_force),
            "climbing_image_idx": ci_idx,
            "path_energy_profile": energies.copy(),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        """运行到收敛或最大步数"""
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self):
        """重置到初始状态"""
        self.images = self._initial_images.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._velocities = np.zeros_like(self.images)
        self._fire_dt = 0.05
        self._fire_alpha = self._fire_alpha_start
        self._fire_steps_since_negative = 0

    def get_trajectory(self) -> list[dict]:
        """获取轨迹"""
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        """是否已收敛"""
        return self._converged

    def get_energy_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """获取能量沿路径的分布

        Returns:
            distances: 沿路径的累积距离
            energies: 各镜像点的能量
        """
        n = self.n_images
        distances = np.zeros(n)
        for i in range(1, n):
            distances[i] = distances[i - 1] + np.linalg.norm(
                self.images[i] - self.images[i - 1]
            )

        energies = np.zeros(n)
        for i in range(n):
            energies[i] = self.pes.energy(self.images[i][0], self.images[i][1])

        return distances, energies


class SteepestDescentMethod:
    """最速下降法（极小值搜索）

    核心迭代公式：q_{t+1} = q_t - α · ∇E(q_t)
    沿负梯度方向搜索，是最基本的优化方法。
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        step_size: float = 0.05,
        conv_threshold: float = 1e-4,
        max_iter: int = 200,
        max_step: float = 0.3,
        use_line_search: bool = True,
    ):
        self.pes = pes
        self.step_size = step_size
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.max_step = max_step
        self.use_line_search = use_line_search

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

    def step(self) -> dict:
        """执行一步最速下降迭代"""
        x, y = self.position

        # 1. 计算梯度
        grad = np.array(self.pes.gradient(x, y), dtype=float)
        grad_norm = np.linalg.norm(grad)

        # 2. 搜索方向：负梯度
        direction = -grad

        # 3. 限制步长
        dir_norm = np.linalg.norm(direction)
        if dir_norm > self.max_step:
            direction = direction * self.max_step / dir_norm

        # 4. 回溯线搜索
        alpha = 1.0
        if self.use_line_search:
            current_energy = self.pes.energy(x, y)
            c1 = 1e-4
            directional_deriv = np.dot(grad, direction)
            for _ in range(20):
                new_pos = self.position + alpha * direction
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy <= current_energy + c1 * alpha * directional_deriv:
                    break
                alpha *= 0.5
        else:
            alpha = self.step_size

        # 5. 更新位置
        self.position = self.position + alpha * direction
        self._iter_count += 1

        # 6. 检查收敛
        self._converged = grad_norm < self.conv_threshold

        new_x, new_y = self.position
        state = {
            "position": self.position.copy(),
            "energy": float(self.pes.energy(new_x, new_y)),
            "gradient": grad.copy(),
            "step_size": float(alpha * np.linalg.norm(direction)),
            "gradient_norm": float(grad_norm),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class BasinHoppingMethod:
    """盆地跳跃法（Basin Hopping）— 全局极小值搜索

    核心原理：
    1. 对当前结构随机扰动
    2. 局部优化（最速下降）到最近极小点
    3. Metropolis准则判断是否接受新极小点

    接受准则：P(accept) = min(1, exp(-(E_new - E_old) / kT))
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        step_size: float = 0.5,
        temperature: float = 1.0,
        conv_threshold: float = 1e-4,
        max_iter: int = 100,
        local_max_iter: int = 50,
    ):
        self.pes = pes
        self.step_size = step_size
        self.temperature = temperature
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(start_pos[0], start_pos[1]))

    def _local_optimize(self, pos: np.ndarray) -> tuple:
        """局部优化：使用最速下降法找到最近极小点"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.conv_threshold:
                break
            # 最速下降步
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            # 简单线搜索
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def step(self) -> dict:
        """执行一步盆地跳跃"""
        # 1. 随机扰动
        perturbation = np.random.randn(2) * self.step_size
        trial_pos = self.position + perturbation

        # 2. 局部优化
        local_pos, local_energy = self._local_optimize(trial_pos)

        # 3. Metropolis准则
        current_energy = self.pes.energy(self.position[0], self.position[1])
        delta_e = local_energy - current_energy

        if delta_e < 0:
            accept = True
        else:
            if self.temperature > 0:
                accept = np.random.random() < np.exp(-delta_e / self.temperature)
            else:
                accept = False

        if accept:
            self.position = local_pos
            current_energy = local_energy

        # 更新全局最优
        if current_energy < self._best_energy:
            self._best_energy = current_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": float(current_energy),
            "trial_position": local_pos.copy(),
            "trial_energy": float(local_energy),
            "accepted": accept,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class MetropolisMCMethod:
    """Metropolis Monte Carlo 方法 — 势能面采样

    核心原理：
    1. 从当前结构按提议分布生成候选结构
    2. 计算接受概率：A = min(1, P(x')/P(x)) = min(1, exp(-(E'-E)/kT))
    3. 按接受概率决定是否接受新结构

    目标分布为玻尔兹曼分布 P(x) ∝ exp(-E(x)/kT)
    """

    def __init__(
        self,
        pes,
        start_pos: np.ndarray,
        step_size: float = 0.5,
        temperature: float = 1.0,
        max_iter: int = 200,
    ):
        self.pes = pes
        self.step_size = step_size
        self.temperature = temperature
        self.max_iter = max_iter

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(start_pos[0], start_pos[1]))
        self._accept_count = 0

    def step(self) -> dict:
        """执行一步Metropolis MC"""
        # 1. 提议步：高斯随机扰动
        perturbation = np.random.randn(2) * self.step_size
        proposed_pos = self.position + perturbation

        # 2. 计算能量差
        current_energy = self.pes.energy(self.position[0], self.position[1])
        proposed_energy = self.pes.energy(proposed_pos[0], proposed_pos[1])
        delta_e = proposed_energy - current_energy

        # 3. 接受概率：A = min(1, exp(-ΔE/kT))
        if delta_e < 0:
            accept = True
        else:
            if self.temperature > 0:
                accept = np.random.random() < np.exp(-delta_e / self.temperature)
            else:
                accept = False

        if accept:
            self.position = proposed_pos
            current_energy = proposed_energy
            self._accept_count += 1

        # 更新全局最优
        if current_energy < self._best_energy:
            self._best_energy = current_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": float(current_energy),
            "proposed_position": proposed_pos.copy(),
            "proposed_energy": float(proposed_energy),
            "accepted": accept,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "accept_rate": float(self._accept_count / self._iter_count),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))
        self._accept_count = 0

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class SSWMethod:
    """随机势能面行走 (Stochastic Surface Walking) — 全局结构搜索

    核心原理：
    1. 随机生成位移方向
    2. 沿该方向添加高斯偏置势"爬山"
    3. 到达一定高度后局部优化，落入新极小点
    4. Metropolis准则判断是否接受新极小点

    高斯偏置势：V_bias(R) = Σ A_i * exp(-|R - R_i|² / (2σ²))
    接受准则：P(accept) = min(1, exp(-(E_new - E_old) / kT))
    """

    def __init__(self, pes, start_pos, step_size=0.5, gaussian_height=5.0,
                 gaussian_width=0.5, temperature=1.0, max_iter=100, local_max_iter=50):
        self.pes = pes
        self.step_size = step_size
        self.gaussian_height = gaussian_height
        self.gaussian_width = gaussian_width
        self.temperature = temperature
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(start_pos[0], start_pos[1]))
        self._gaussians: list[tuple] = []
        self._current_bias = None

    def _compute_bias_energy(self, x, y):
        """Compute total bias potential at (x,y)"""
        total = 0.0
        for center, height, width in self._gaussians:
            dist_sq = (x - center[0])**2 + (y - center[1])**2
            total += height * np.exp(-dist_sq / (2 * width**2))
        return total

    def _compute_bias_gradient(self, x, y):
        """Compute gradient of total bias potential at (x,y)"""
        grad = np.zeros(2)
        for center, height, width in self._gaussians:
            dist_sq = (x - center[0])**2 + (y - center[1])**2
            exp_val = height * np.exp(-dist_sq / (2 * width**2))
            grad[0] += exp_val * (-(x - center[0]) / (width**2))
            grad[1] += exp_val * (-(y - center[1]) / (width**2))
        return grad

    def get_bias_function(self):
        """Return the current bias function for visualization"""
        if not self._gaussians:
            return None

        def bias_func(x, y):
            return self._compute_bias_energy(x, y)
        return bias_func

    def _local_optimize_biased(self, pos):
        """Local optimization on biased PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            pes_grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            bias_grad = self._compute_bias_gradient(current_pos[0], current_pos[1])
            total_grad = pes_grad + bias_grad
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm < 1e-4:
                break
            step = -total_grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_total_e = self.pes.energy(current_pos[0], current_pos[1]) + \
                self._compute_bias_energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_total_e = self.pes.energy(new_pos[0], new_pos[1]) + \
                    self._compute_bias_energy(new_pos[0], new_pos[1])
                if new_total_e < current_total_e:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos

    def _local_optimize(self, pos):
        """Local optimization on true PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def step(self):
        """执行一步SSW迭代"""
        # 1. Generate random displacement direction
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)

        # 2. Add Gaussian bias along that direction
        bias_center = self.position + direction * self.step_size
        self._gaussians.append((bias_center.copy(), self.gaussian_height, self.gaussian_width))

        # 3. Local optimize on biased PES
        biased_pos = self._local_optimize_biased(self.position)

        # 4. Remove bias, local optimize on true PES
        new_pos, new_energy = self._local_optimize(biased_pos)

        # 5. Metropolis criterion
        current_energy = float(self.pes.energy(self.position[0], self.position[1]))
        delta_e = new_energy - current_energy

        if delta_e < 0:
            accept = True
        else:
            if self.temperature > 0:
                accept = np.random.random() < np.exp(-delta_e / self.temperature)
            else:
                accept = False

        if accept:
            self.position = new_pos
            current_energy = new_energy

        # Update global best
        if current_energy < self._best_energy:
            self._best_energy = current_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        # 6. Record state with bias info
        bias_gaussians_copy = [(c.copy(), h, w) for c, h, w in self._gaussians]
        state = {
            "position": self.position.copy(),
            "energy": float(current_energy),
            "bias_gaussians": bias_gaussians_copy,
            "accepted": accept,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))
        self._gaussians = []
        self._current_bias = None

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class MetaDynamicsMethod:
    """元动力学 (MetaDynamics) — 增强采样方法

    核心原理：
    在反应坐标上持续添加高斯型排斥偏置势，驱使体系逃离局部极小点
    偏置势：V_bias(ξ, t) = Σ w * exp(-(ξ - ξ(t'))² / (2δξ²))
    """

    def __init__(self, pes, start_pos, gaussian_height=1.0, gaussian_width=0.3,
                 max_iter=200, local_max_iter=20, add_interval=1):
        self.pes = pes
        self.gaussian_height = gaussian_height
        self.gaussian_width = gaussian_width
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter
        self.add_interval = add_interval

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(start_pos[0], start_pos[1]))
        self._gaussians: list[tuple] = []

    def _compute_bias_energy(self, x, y):
        """Compute total bias potential at (x,y)"""
        total = 0.0
        for center, height, width in self._gaussians:
            dist_sq = (x - center[0])**2 + (y - center[1])**2
            total += height * np.exp(-dist_sq / (2 * width**2))
        return total

    def _compute_bias_gradient(self, x, y):
        """Compute gradient of total bias potential at (x,y)"""
        grad = np.zeros(2)
        for center, height, width in self._gaussians:
            dist_sq = (x - center[0])**2 + (y - center[1])**2
            exp_val = height * np.exp(-dist_sq / (2 * width**2))
            grad[0] += exp_val * (-(x - center[0]) / (width**2))
            grad[1] += exp_val * (-(y - center[1]) / (width**2))
        return grad

    def get_bias_function(self):
        """Return current bias function for visualization"""
        if not self._gaussians:
            return None

        def bias_func(x, y):
            return self._compute_bias_energy(x, y)
        return bias_func

    def _local_optimize(self, pos):
        """Local optimization on true PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def _local_optimize_biased(self, pos):
        """Local optimization on biased PES to escape current well"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            pes_grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            bias_grad = self._compute_bias_gradient(current_pos[0], current_pos[1])
            total_grad = pes_grad + bias_grad
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm < 1e-4:
                break
            step = -total_grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_total_e = self.pes.energy(current_pos[0], current_pos[1]) + \
                self._compute_bias_energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_total_e = self.pes.energy(new_pos[0], new_pos[1]) + \
                    self._compute_bias_energy(new_pos[0], new_pos[1])
                if new_total_e < current_total_e:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos

    def step(self):
        """执行一步元动力学迭代"""
        # 1. Local optimize from current position
        local_pos, local_energy = self._local_optimize(self.position)

        # 2. Add Gaussian at current minimum position (at interval)
        if self._iter_count % self.add_interval == 0:
            self._gaussians.append((local_pos.copy(), self.gaussian_height, self.gaussian_width))

        # 3. On biased PES, move to escape current well
        escaped_pos = self._local_optimize_biased(local_pos)

        # 4. Local optimize on true PES from escaped position
        new_pos, new_energy = self._local_optimize(escaped_pos)

        # Update position
        self.position = new_pos

        # Update global best
        if new_energy < self._best_energy:
            self._best_energy = new_energy
            self._best_pos = new_pos.copy()

        self._iter_count += 1

        # Record state with bias info
        bias_gaussians_copy = [(c.copy(), h, w) for c, h, w in self._gaussians]
        state = {
            "position": self.position.copy(),
            "energy": float(new_energy),
            "bias_gaussians": bias_gaussians_copy,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))
        self._gaussians = []

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class MinimaHoppingMethod:
    """极小点跳跃 (Minima Hopping) — 全局搜索

    核心原理：
    通过短时间分子动力学逃离当前极小点，随后局部优化得到新极小点
    动态调整动能和接受阈值
    """

    def __init__(self, pes, start_pos, initial_kinetic=0.5, max_iter=100,
                 local_max_iter=50, md_steps=10, md_dt=0.01):
        self.pes = pes
        self.initial_kinetic = initial_kinetic
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter
        self.md_steps = md_steps
        self.md_dt = md_dt

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(start_pos[0], start_pos[1]))
        self._kinetic_energy = initial_kinetic
        self._accept_threshold = 0.0
        self._current_energy = float(pes.energy(start_pos[0], start_pos[1]))

    def _local_optimize(self, pos):
        """Local optimization on true PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def _run_md(self, pos, kinetic_energy):
        """Run short MD simulation using velocity Verlet on PES"""
        current_pos = pos.copy()
        # Initialize velocities from Maxwell-Boltzmann-like distribution
        speed = np.sqrt(2.0 * kinetic_energy)
        random_dir = np.random.randn(2)
        random_dir = random_dir / np.linalg.norm(random_dir)
        velocity = speed * random_dir

        # Velocity Verlet integration
        for _ in range(self.md_steps):
            x, y = current_pos
            force = -np.array(self.pes.gradient(x, y), dtype=float)
            # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
            new_pos = current_pos + velocity * self.md_dt + 0.5 * force * self.md_dt**2
            # Compute new force
            new_force = -np.array(self.pes.gradient(new_pos[0], new_pos[1]), dtype=float)
            # Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            velocity = velocity + 0.5 * (force + new_force) * self.md_dt
            current_pos = new_pos

        return current_pos

    def step(self):
        """执行一步极小点跳跃"""
        # 1. Run short MD simulation to escape current minimum
        md_end_pos = self._run_md(self.position, self._kinetic_energy)

        # 2. Local optimize from MD endpoint
        new_pos, new_energy = self._local_optimize(md_end_pos)

        # 3. Accept/reject based on energy difference with adaptive threshold
        delta_e = new_energy - self._current_energy

        if delta_e < self._accept_threshold:
            accept = True
        else:
            accept = False

        # 4. Adjust kinetic energy and acceptance threshold
        if accept:
            self.position = new_pos
            self._current_energy = new_energy
            # Decrease kinetic energy (easier to accept next time)
            self._kinetic_energy *= 0.8
            # Decrease acceptance threshold
            self._accept_threshold = min(self._accept_threshold, delta_e) - 0.1
        else:
            # Increase kinetic energy (harder to escape, need more energy)
            self._kinetic_energy *= 1.2
            # Increase acceptance threshold
            self._accept_threshold += 0.1

        # Update global best
        if self._current_energy < self._best_energy:
            self._best_energy = self._current_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": float(self._current_energy),
            "accepted": accept,
            "kinetic_energy": float(self._kinetic_energy),
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))
        self._kinetic_energy = self.initial_kinetic
        self._accept_threshold = 0.0
        self._current_energy = float(self.pes.energy(self.position[0], self.position[1]))

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_bias_function(self):
        """MinimaHopping does not use bias potential"""
        return None


class GeneticAlgorithmMethod:
    """遗传算法 (Genetic Algorithm) — 全局搜索

    核心原理：
    1. 初始化随机种群
    2. 选择优良个体
    3. 交叉产生子代
    4. 变异
    5. 局部优化
    """

    def __init__(self, pes, start_pos, population_size=20, mutation_rate=0.3,
                 max_iter=100, local_max_iter=20, search_range=2.0):
        self.pes = pes
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter
        self.search_range = search_range

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # Initialize population around start_pos
        self.population = []
        for _ in range(population_size):
            individual = self.position + np.random.randn(2) * search_range * 0.5
            energy = float(pes.energy(individual[0], individual[1]))
            self.population.append((individual.copy(), energy))

        # Sort by energy
        self.population.sort(key=lambda x: x[1])

        self._best_pos = self.population[0][0].copy()
        self._best_energy = self.population[0][1]

    def _local_optimize(self, pos):
        """Local optimization on true PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def _tournament_selection(self, tournament_size=3):
        """Tournament selection"""
        candidates = np.random.choice(len(self.population), min(tournament_size, len(self.population)), replace=False)
        best_idx = min(candidates, key=lambda i: self.population[i][1])
        return self.population[best_idx]

    def _blend_crossover(self, parent1, parent2, alpha=0.5):
        """Blend crossover (BLX-alpha)"""
        child = np.zeros(2)
        for d in range(2):
            low = min(parent1[d], parent2[d])
            high = max(parent1[d], parent2[d])
            diff = high - low
            child[d] = np.random.uniform(low - alpha * diff, high + alpha * diff)
        return child

    def _mutate(self, individual):
        """Gaussian mutation"""
        if np.random.random() < self.mutation_rate:
            individual = individual + np.random.randn(2) * self.search_range * 0.1
        return individual

    def step(self):
        """执行一步遗传算法迭代"""
        n_offspring = max(2, self.population_size // 2)
        offspring = []

        # 1. Selection, crossover, mutation, local optimization
        for _ in range(n_offspring):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            child_pos = self._blend_crossover(parent1[0], parent2[0])
            child_pos = self._mutate(child_pos)
            # Local optimize
            child_pos, child_energy = self._local_optimize(child_pos)
            offspring.append((child_pos.copy(), child_energy))

        # 2. Replace worst individuals
        combined = self.population + offspring
        combined.sort(key=lambda x: x[1])
        self.population = combined[:self.population_size]

        # Update best
        if self.population[0][1] < self._best_energy:
            self._best_energy = self.population[0][1]
            self._best_pos = self.population[0][0].copy()

        self.position = self._best_pos.copy()
        self._iter_count += 1

        # State includes population positions for visualization
        population_positions = [ind[0].copy() for ind in self.population]
        state = {
            "position": self.position.copy(),
            "energy": float(self._best_energy),
            "population_positions": population_positions,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

        # Reinitialize population
        self.population = []
        for _ in range(self.population_size):
            individual = self.position + np.random.randn(2) * self.search_range * 0.5
            energy = float(self.pes.energy(individual[0], individual[1]))
            self.population.append((individual.copy(), energy))
        self.population.sort(key=lambda x: x[1])
        self._best_pos = self.population[0][0].copy()
        self._best_energy = self.population[0][1]

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_bias_function(self):
        """GeneticAlgorithm does not use bias potential"""
        return None


class PSOMethod:
    """粒子群优化 (Particle Swarm Optimization) — 全局搜索

    核心原理：
    每个粒子根据自身速度、历史最优和全局最优更新位置
    v_i = w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
    x_i = x_i + v_i
    """

    def __init__(self, pes, start_pos, n_particles=20, w=0.7, c1=1.5, c2=1.5,
                 max_iter=100, local_max_iter=20, search_range=2.0):
        self.pes = pes
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.local_max_iter = local_max_iter
        self.search_range = search_range

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # Initialize particles around start_pos
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_energies = []

        for _ in range(n_particles):
            particle = self.position + np.random.randn(2) * search_range * 0.5
            self.particles.append(particle.copy())
            self.velocities.append(np.random.randn(2) * 0.1)
            energy = float(pes.energy(particle[0], particle[1]))
            self.personal_best.append(particle.copy())
            self.personal_best_energies.append(energy)

        # Global best
        best_idx = int(np.argmin(self.personal_best_energies))
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_energy = self.personal_best_energies[best_idx]

        self._best_pos = self.global_best.copy()
        self._best_energy = self.global_best_energy

    def _local_optimize(self, pos):
        """Local optimization on true PES"""
        current_pos = pos.copy()
        for _ in range(self.local_max_iter):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def step(self):
        """执行一步PSO迭代"""
        # 1. Update velocities and positions
        for i in range(self.n_particles):
            r1 = np.random.random(2)
            r2 = np.random.random(2)

            cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
            social = self.c2 * r2 * (self.global_best - self.particles[i])

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Clamp velocity
            v_norm = np.linalg.norm(self.velocities[i])
            max_v = self.search_range * 0.5
            if v_norm > max_v:
                self.velocities[i] = self.velocities[i] * max_v / v_norm

            self.particles[i] = self.particles[i] + self.velocities[i]

        # 2. Local optimize each particle (periodically, every few iterations)
        if self._iter_count % 5 == 0:
            for i in range(self.n_particles):
                opt_pos, opt_energy = self._local_optimize(self.particles[i])
                self.particles[i] = opt_pos

        # 3. Update personal best and global best
        for i in range(self.n_particles):
            energy = float(self.pes.energy(self.particles[i][0], self.particles[i][1]))
            if energy < self.personal_best_energies[i]:
                self.personal_best_energies[i] = energy
                self.personal_best[i] = self.particles[i].copy()

        best_idx = int(np.argmin(self.personal_best_energies))
        if self.personal_best_energies[best_idx] < self.global_best_energy:
            self.global_best_energy = self.personal_best_energies[best_idx]
            self.global_best = self.personal_best[best_idx].copy()

        # Update best
        if self.global_best_energy < self._best_energy:
            self._best_energy = self.global_best_energy
            self._best_pos = self.global_best.copy()

        self.position = self._best_pos.copy()
        self._iter_count += 1

        # State includes particle positions for visualization
        particle_positions = [p.copy() for p in self.particles]
        state = {
            "position": self.position.copy(),
            "energy": float(self._best_energy),
            "particle_positions": particle_positions,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

        # Reinitialize particles
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_energies = []

        for _ in range(self.n_particles):
            particle = self.position + np.random.randn(2) * self.search_range * 0.5
            self.particles.append(particle.copy())
            self.velocities.append(np.random.randn(2) * 0.1)
            energy = float(self.pes.energy(particle[0], particle[1]))
            self.personal_best.append(particle.copy())
            self.personal_best_energies.append(energy)

        best_idx = int(np.argmin(self.personal_best_energies))
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_energy = self.personal_best_energies[best_idx]
        self._best_pos = self.global_best.copy()
        self._best_energy = self.global_best_energy

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_bias_function(self):
        """PSO does not use bias potential"""
        return None


class ABCMethod:
    """人工蜂群算法 (Artificial Bee Colony) — 全局搜索

    模拟三类蜜蜂协作：雇佣蜂在已知食物源附近搜索；
    观察蜂按食物源质量选择跟随；侦察蜂随机探索新区域。
    """

    def __init__(self, pes, start_pos, n_bees=20, limit=50, max_iter=100, search_range=2.0):
        self.pes = pes
        self.n_bees = n_bees
        self.limit = limit
        self.max_iter = max_iter
        self.search_range = search_range

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # 初始化食物源
        self.food_sources = []
        self.food_energies = []
        self.trial_counts = []

        for _ in range(n_bees):
            source = self.position + np.random.randn(2) * search_range * 0.5
            energy = float(pes.energy(source[0], source[1]))
            self.food_sources.append(source.copy())
            self.food_energies.append(energy)
            self.trial_counts.append(0)

        best_idx = int(np.argmin(self.food_energies))
        self._best_pos = self.food_sources[best_idx].copy()
        self._best_energy = self.food_energies[best_idx]

    def _local_optimize(self, pos):
        """局部优化"""
        current_pos = pos.copy()
        for _ in range(20):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def step(self):
        """执行一步ABC迭代"""
        # 1. 雇佣蜂阶段
        for i in range(self.n_bees):
            k = np.random.randint(self.n_bees)
            while k == i:
                k = np.random.randint(self.n_bees)
            phi = np.random.randn(2) * 0.5
            new_source = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[k])
            new_energy = float(self.pes.energy(new_source[0], new_source[1]))

            if new_energy < self.food_energies[i]:
                self.food_sources[i] = new_source
                self.food_energies[i] = new_energy
                self.trial_counts[i] = 0
            else:
                self.trial_counts[i] += 1

        # 2. 观察蜂阶段
        max_energy = max(self.food_energies)
        fitness = [max_energy - e + 1e-10 for e in self.food_energies]
        total_fitness = sum(fitness)
        if total_fitness > 0:
            probs = [f / total_fitness for f in fitness]
        else:
            probs = [1.0 / self.n_bees] * self.n_bees

        for _ in range(self.n_bees):
            r = np.random.random()
            cumsum = 0
            selected = 0
            for j in range(self.n_bees):
                cumsum += probs[j]
                if r <= cumsum:
                    selected = j
                    break

            k = np.random.randint(self.n_bees)
            while k == selected:
                k = np.random.randint(self.n_bees)
            phi = np.random.randn(2) * 0.5
            new_source = self.food_sources[selected] + phi * (self.food_sources[selected] - self.food_sources[k])
            new_energy = float(self.pes.energy(new_source[0], new_source[1]))

            if new_energy < self.food_energies[selected]:
                self.food_sources[selected] = new_source
                self.food_energies[selected] = new_energy
                self.trial_counts[selected] = 0
            else:
                self.trial_counts[selected] += 1

        # 3. 侦察蜂阶段
        for i in range(self.n_bees):
            if self.trial_counts[i] > self.limit:
                new_source = self.position + np.random.randn(2) * self.search_range
                new_energy = float(self.pes.energy(new_source[0], new_source[1]))
                self.food_sources[i] = new_source
                self.food_energies[i] = new_energy
                self.trial_counts[i] = 0

        # 4. 定期局部优化最优个体
        if self._iter_count % 5 == 0:
            best_idx = int(np.argmin(self.food_energies))
            opt_pos, opt_energy = self._local_optimize(self.food_sources[best_idx])
            self.food_sources[best_idx] = opt_pos
            self.food_energies[best_idx] = opt_energy

        # 更新最优
        best_idx = int(np.argmin(self.food_energies))
        if self.food_energies[best_idx] < self._best_energy:
            self._best_energy = self.food_energies[best_idx]
            self._best_pos = self.food_sources[best_idx].copy()

        self.position = self._best_pos.copy()
        self._iter_count += 1

        population_positions = [s.copy() for s in self.food_sources]
        state = {
            "position": self.position.copy(),
            "energy": float(self._best_energy),
            "population_positions": population_positions,
            "best_position": self._best_pos.copy(),
            "best_energy": float(self._best_energy),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0

        self.food_sources = []
        self.food_energies = []
        self.trial_counts = []
        for _ in range(self.n_bees):
            source = self.position + np.random.randn(2) * self.search_range * 0.5
            energy = float(self.pes.energy(source[0], source[1]))
            self.food_sources.append(source.copy())
            self.food_energies.append(energy)
            self.trial_counts.append(0)

        best_idx = int(np.argmin(self.food_energies))
        self._best_pos = self.food_sources[best_idx].copy()
        self._best_energy = self.food_energies[best_idx]

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_bias_function(self):
        """ABC does not use bias potential"""
        return None


class UmbrellaSamplingMethod:
    """伞形采样 (Umbrella Sampling) — 增强采样

    在反应坐标上设置一系列谐振偏置窗口，将体系限制在特定区间采样。
    V_bias(ξ) = ½ k (ξ - ξ₀)²
    通过加权直方图分析(WHAM)合并各窗口数据，得到完整自由能曲线。
    """

    def __init__(self, pes, start_pos, end_pos, n_windows=5, spring_k=10.0,
                 mc_step_size=0.1, mc_temperature=1.0, steps_per_window=20, max_iter=100):
        self.pes = pes
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.n_windows = n_windows
        self.spring_k = spring_k
        self.mc_step_size = mc_step_size
        self.mc_temperature = mc_temperature
        self.steps_per_window = steps_per_window
        self.max_iter = max_iter

        # 反应坐标：从起点到终点的直线
        self.rc_direction = self.end_pos - self.start_pos
        self.rc_length = np.linalg.norm(self.rc_direction)
        if self.rc_length > 1e-10:
            self.rc_direction = self.rc_direction / self.rc_length
        else:
            self.rc_direction = np.array([1.0, 0.0])
            self.rc_length = 1.0

        # 窗口中心沿反应坐标分布
        self.window_centers = []
        for i in range(n_windows):
            t = i / max(n_windows - 1, 1)
            center = self.start_pos + t * (self.end_pos - self.start_pos)
            self.window_centers.append(center)

        # 当前状态
        self.current_window = 0
        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # 当前偏置窗口中心
        self._current_window_center = self.window_centers[0].copy()

        # 最优解
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(self.position[0], self.position[1]))

    def _project_to_rc(self, pos):
        """将位置投影到反应坐标上，返回标量值"""
        return np.dot(pos - self.start_pos, self.rc_direction)

    def _compute_bias_energy(self, x, y):
        """计算当前窗口的谐振偏置势能"""
        pos = np.array([x, y])
        xi = self._project_to_rc(pos)
        xi_0 = self._project_to_rc(self._current_window_center)
        return 0.5 * self.spring_k * (xi - xi_0) ** 2

    def _compute_bias_gradient(self, x, y):
        """计算当前窗口的谐振偏置梯度"""
        pos = np.array([x, y])
        xi = self._project_to_rc(pos)
        xi_0 = self._project_to_rc(self._current_window_center)
        dV_dxi = self.spring_k * (xi - xi_0)
        return dV_dxi * self.rc_direction

    def get_bias_function(self):
        """返回当前偏置函数用于可视化"""
        center = self._current_window_center.copy()
        spring_k = self.spring_k
        start_pos = self.start_pos.copy()
        rc_dir = self.rc_direction.copy()

        def bias_func(x, y):
            pos = np.array([x, y])
            xi = np.dot(pos - start_pos, rc_dir)
            xi_0 = np.dot(center - start_pos, rc_dir)
            return 0.5 * spring_k * (xi - xi_0) ** 2

        return bias_func

    def step(self):
        """执行一步伞形采样"""
        # 如果当前窗口采样完成，切换到下一个窗口
        if self._iter_count > 0 and self._iter_count % self.steps_per_window == 0:
            self.current_window += 1
            if self.current_window >= self.n_windows:
                self._converged = True
                state = {
                    "position": self.position.copy(),
                    "energy": float(self.pes.energy(self.position[0], self.position[1])),
                    "current_window": self.current_window,
                    "window_center": self._current_window_center.copy(),
                    "iteration": self._iter_count,
                }
                self._trajectory.append(state)
                return state

            # 移动到新窗口中心
            self._current_window_center = self.window_centers[self.current_window].copy()
            self.position = self._current_window_center.copy()

        # Metropolis MC步，带谐振偏置
        proposal = self.position + np.random.randn(2) * self.mc_step_size

        current_energy = float(self.pes.energy(self.position[0], self.position[1]))
        current_bias = self._compute_bias_energy(self.position[0], self.position[1])
        current_total = current_energy + current_bias

        proposal_energy = float(self.pes.energy(proposal[0], proposal[1]))
        proposal_bias = self._compute_bias_energy(proposal[0], proposal[1])
        proposal_total = proposal_energy + proposal_bias

        delta_E = proposal_total - current_total

        if delta_E < 0 or np.random.random() < np.exp(-delta_E / max(self.mc_temperature, 1e-10)):
            self.position = proposal

        # 更新最优
        pos_energy = float(self.pes.energy(self.position[0], self.position[1]))
        if pos_energy < self._best_energy:
            self._best_energy = pos_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": pos_energy,
            "current_window": self.current_window,
            "window_center": self._current_window_center.copy(),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None, end_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.start_pos = np.array(start_pos, dtype=float)
        if end_pos is not None:
            self.end_pos = np.array(end_pos, dtype=float)

        self.rc_direction = self.end_pos - self.start_pos
        self.rc_length = np.linalg.norm(self.rc_direction)
        if self.rc_length > 1e-10:
            self.rc_direction = self.rc_direction / self.rc_length
        else:
            self.rc_direction = np.array([1.0, 0.0])
            self.rc_length = 1.0

        self.window_centers = []
        for i in range(self.n_windows):
            t = i / max(self.n_windows - 1, 1)
            center = self.start_pos + t * (self.end_pos - self.start_pos)
            self.window_centers.append(center)

        self.current_window = 0
        self.position = self.start_pos.copy()
        self._start_pos = self.position.copy()
        self._current_window_center = self.window_centers[0].copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class ABFMethod:
    """自适应偏置力法 (Adaptive Biasing Force) — 增强采样

    沿反应坐标逐步计算平均力，将其负值作为偏置力施加到体系，
    抵消自由能梯度；最终体系沿反应坐标自由扩散。
    V_bias(ξ) = -∫ F_avg(ξ') dξ'
    """

    def __init__(self, pes, start_pos, end_pos, n_bins=20, max_iter=200,
                 mc_step_size=0.1, mc_temperature=1.0):
        self.pes = pes
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.mc_step_size = mc_step_size
        self.mc_temperature = mc_temperature

        # 反应坐标
        self.rc_direction = self.end_pos - self.start_pos
        self.rc_length = np.linalg.norm(self.rc_direction)
        if self.rc_length > 1e-10:
            self.rc_direction = self.rc_direction / self.rc_length
        else:
            self.rc_direction = np.array([1.0, 0.0])
            self.rc_length = 1.0

        # 沿反应坐标的分箱
        self.bin_edges = np.linspace(0, self.rc_length, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # 累积力和计数
        self.bin_forces = np.zeros(n_bins)
        self.bin_counts = np.zeros(n_bins)

        # 当前状态
        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # 最优解
        self._best_pos = self.position.copy()
        self._best_energy = float(pes.energy(self.position[0], self.position[1]))

    def _project_to_rc(self, pos):
        """将位置投影到反应坐标"""
        return np.dot(pos - self.start_pos, self.rc_direction)

    def _get_bin(self, xi):
        """获取反应坐标值对应的bin索引"""
        idx = int((xi / self.rc_length) * self.n_bins)
        return max(0, min(self.n_bins - 1, idx))

    def _compute_bias_energy(self, x, y):
        """计算ABF偏置势能"""
        pos = np.array([x, y])
        xi = self._project_to_rc(pos)
        bin_idx = self._get_bin(xi)
        bias = 0.0
        for i in range(bin_idx):
            if self.bin_counts[i] > 0:
                avg_force = self.bin_forces[i] / self.bin_counts[i]
                bias -= avg_force * (self.bin_centers[min(i + 1, self.n_bins - 1)] - self.bin_centers[i])
        return bias

    def _compute_bias_gradient(self, x, y):
        """计算ABF偏置梯度"""
        pos = np.array([x, y])
        xi = self._project_to_rc(pos)
        bin_idx = self._get_bin(xi)

        if self.bin_counts[bin_idx] > 0:
            avg_force = self.bin_forces[bin_idx] / self.bin_counts[bin_idx]
            return -avg_force * self.rc_direction
        return np.zeros(2)

    def get_bias_function(self):
        """返回当前偏置函数用于可视化"""
        bin_forces = self.bin_forces.copy()
        bin_counts = self.bin_counts.copy()
        bin_centers = self.bin_centers.copy()
        n_bins = self.n_bins
        rc_length = self.rc_length
        start_pos = self.start_pos.copy()
        rc_dir = self.rc_direction.copy()

        def bias_func(x, y):
            pos = np.array([x, y])
            xi = np.dot(pos - start_pos, rc_dir)
            bin_idx = max(0, min(n_bins - 1, int(xi / rc_length * n_bins)))
            bias = 0.0
            for i in range(bin_idx):
                if bin_counts[i] > 0:
                    avg_force = bin_forces[i] / bin_counts[i]
                    bias -= avg_force * (bin_centers[min(i + 1, n_bins - 1)] - bin_centers[i])
            return bias

        return bias_func

    def step(self):
        """执行一步ABF采样"""
        # Metropolis MC步，带ABF偏置
        proposal = self.position + np.random.randn(2) * self.mc_step_size

        current_energy = float(self.pes.energy(self.position[0], self.position[1]))
        current_bias = self._compute_bias_energy(self.position[0], self.position[1])
        current_total = current_energy + current_bias

        proposal_energy = float(self.pes.energy(proposal[0], proposal[1]))
        proposal_bias = self._compute_bias_energy(proposal[0], proposal[1])
        proposal_total = proposal_energy + proposal_bias

        delta_E = proposal_total - current_total

        if delta_E < 0 or np.random.random() < np.exp(-delta_E / max(self.mc_temperature, 1e-10)):
            self.position = proposal

        # 累积沿反应坐标的力
        xi = self._project_to_rc(self.position)
        if 0 <= xi <= self.rc_length:
            bin_idx = self._get_bin(xi)
            grad = np.array(self.pes.gradient(self.position[0], self.position[1]), dtype=float)
            force_along_rc = -np.dot(grad, self.rc_direction)
            self.bin_forces[bin_idx] += force_along_rc
            self.bin_counts[bin_idx] += 1

        # 更新最优
        pos_energy = float(self.pes.energy(self.position[0], self.position[1]))
        if pos_energy < self._best_energy:
            self._best_energy = pos_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": pos_energy,
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None, end_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.start_pos = np.array(start_pos, dtype=float)
        if end_pos is not None:
            self.end_pos = np.array(end_pos, dtype=float)

        self.rc_direction = self.end_pos - self.start_pos
        self.rc_length = np.linalg.norm(self.rc_direction)
        if self.rc_length > 1e-10:
            self.rc_direction = self.rc_direction / self.rc_length
        else:
            self.rc_direction = np.array([1.0, 0.0])
            self.rc_length = 1.0

        self.bin_edges = np.linspace(0, self.rc_length, self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_forces = np.zeros(self.n_bins)
        self.bin_counts = np.zeros(self.n_bins)

        self.position = self.start_pos.copy()
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._best_pos = self.position.copy()
        self._best_energy = float(self.pes.energy(self.position[0], self.position[1]))

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged


class CBDMethod:
    """约束布罗伊登双子法 (Constrained Broyden Dimer) — 过渡态搜索

    构造双子结构，通过数值二阶导数确定最低曲率方向（虚频方向）；
    沿该方向爬升的同时，在垂直方向做能量极小化，最终收敛到鞍点。
    使用Broyden更新代替精确Hessian计算。

    有效优化力：F_eff = F⊥ - λ·N̂
    其中 λ = (F₁-F₂)·N̂ / ΔR 为曲率
    """

    def __init__(self, pes, start_pos, delta_r=0.01, conv_threshold=1e-3,
                 max_iter=100, trans_step=0.1):
        self.pes = pes
        self.delta_r = delta_r
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.trans_step = trans_step

        self.position = np.array(start_pos, dtype=float)
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # 初始化双子方向
        self.direction = np.array([1.0, 0.0])

        # Broyden矩阵用于旋转更新
        self._B = np.eye(2)

    def step(self):
        """执行一步CBD迭代"""
        center = self.position
        N = self.direction / np.linalg.norm(self.direction)

        # 计算双子两端点的力
        R1 = center + self.delta_r * N
        R2 = center - self.delta_r * N

        F1 = -np.array(self.pes.gradient(R1[0], R1[1]), dtype=float)
        F2 = -np.array(self.pes.gradient(R2[0], R2[1]), dtype=float)
        F_center = -np.array(self.pes.gradient(center[0], center[1]), dtype=float)

        # 沿双子方向的曲率
        C = np.dot(F1 - F2, N) / (2 * self.delta_r)

        # 旋转：寻找最低曲率方向
        F_rot = (F1 - F2) - np.dot(F1 - F2, N) * N
        if np.linalg.norm(F_rot) > 1e-10:
            dN = -0.5 * self.delta_r * F_rot / np.linalg.norm(F_rot)
            dN_norm = np.linalg.norm(dN)
            if dN_norm > 0.1:
                dN = dN * 0.1 / dN_norm

            new_N = N + dN
            new_N = new_N / np.linalg.norm(new_N)

            # Broyden更新
            y = F_rot
            s = dN
            if np.linalg.norm(s) > 1e-12:
                self._B = self._B + np.outer(y - self._B @ s, s) / np.dot(s, s)

            self.direction = new_N
            N = new_N

        # 平动：有效力
        F_perp = F_center - np.dot(F_center, N) * N
        F_eff = F_perp - C * N

        step = F_eff
        step_norm = np.linalg.norm(step)
        if step_norm > self.trans_step:
            step = step * self.trans_step / step_norm

        # 回溯线搜索
        alpha = 1.0
        current_energy = float(self.pes.energy(center[0], center[1]))
        for _ in range(10):
            new_pos = center + alpha * step
            new_energy = float(self.pes.energy(new_pos[0], new_pos[1]))
            if new_energy < current_energy + 0.1 * max(abs(current_energy), 1e-6):
                break
            alpha *= 0.5

        self.position = center + alpha * step

        # 检查收敛
        grad = np.array(self.pes.gradient(self.position[0], self.position[1]), dtype=float)
        if np.linalg.norm(grad) < self.conv_threshold:
            self._converged = True

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": float(self.pes.energy(self.position[0], self.position[1])),
            "direction": self.direction.copy(),
            "curvature": C,
            "center": self.position.copy(),
            "gradient_norm": np.linalg.norm(grad),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.position = np.array(start_pos, dtype=float)
            self._start_pos = self.position.copy()
        else:
            self.position = self._start_pos.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self.direction = np.array([1.0, 0.0])
        self._B = np.eye(2)

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged

    def get_bias_function(self):
        """CBD does not use bias potential"""
        return None


class DESWMethod:
    """双端行走法 (Double-End Surface Walking) — 双端路径搜索

    分别从初态和终态出发，沿反应方向添加高斯偏置势并局部优化，
    直到两端轨迹相遇；取轨迹最高点作为过渡态初始猜测。
    偏置势形式与SSW相同：V_bias(R) = Σ Aᵢ·exp(-|R-Rᵢ|²/(2σ²))
    """

    def __init__(self, pes, start_pos, end_pos, step_size=0.5,
                 gaussian_height=5.0, gaussian_width=0.5,
                 temperature=1.0, max_iter=100):
        self.pes = pes
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.step_size = step_size
        self.gaussian_height = gaussian_height
        self.gaussian_width = gaussian_width
        self.temperature = temperature
        self.max_iter = max_iter

        # 两个行走器
        self.position_forward = self.start_pos.copy()
        self.position_backward = self.end_pos.copy()

        # 两端的高斯偏置
        self._gaussians_forward: list[tuple] = []
        self._gaussians_backward: list[tuple] = []

        # 当前位置
        self.position = self.start_pos.copy()
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory: list[dict] = []
        self._iter_count = 0

        # 两端路径
        self._path_forward = [self.start_pos.copy()]
        self._path_backward = [self.end_pos.copy()]

        # 最优
        self._best_pos = self.start_pos.copy()
        self._best_energy = float(pes.energy(self.start_pos[0], self.start_pos[1]))

    def _compute_bias_energy(self, x, y):
        """计算两端的总偏置势能"""
        total = 0.0
        for center, height, width in self._gaussians_forward + self._gaussians_backward:
            dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
            total += height * np.exp(-dist_sq / (2 * width ** 2))
        return total

    def _compute_bias_gradient(self, x, y):
        """计算偏置梯度"""
        total = np.zeros(2)
        for center, height, width in self._gaussians_forward + self._gaussians_backward:
            dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
            exp_val = height * np.exp(-dist_sq / (2 * width ** 2))
            total[0] += exp_val * (-(x - center[0]) / width ** 2)
            total[1] += exp_val * (-(y - center[1]) / width ** 2)
        return total

    def get_bias_function(self):
        """返回当前偏置函数用于可视化"""
        gaussians = list(self._gaussians_forward) + list(self._gaussians_backward)
        if not gaussians:
            return None

        def bias_func(x, y):
            total = 0.0
            for center, height, width in gaussians:
                dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
                total += height * np.exp(-dist_sq / (2 * width ** 2))
            return total

        return bias_func

    def _local_optimize(self, pos):
        """在真实势能面上局部优化"""
        current_pos = pos.copy()
        for _ in range(20):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                break
            step = -grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_energy = self.pes.energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_energy = self.pes.energy(new_pos[0], new_pos[1])
                if new_energy < current_energy:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos, float(self.pes.energy(current_pos[0], current_pos[1]))

    def _local_optimize_biased(self, pos):
        """在偏置势能面上局部优化"""
        current_pos = pos.copy()
        for _ in range(20):
            grad = np.array(self.pes.gradient(current_pos[0], current_pos[1]), dtype=float)
            bias_grad = self._compute_bias_gradient(current_pos[0], current_pos[1])
            total_grad = grad + bias_grad
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm < 1e-4:
                break
            step = -total_grad
            step_norm = np.linalg.norm(step)
            if step_norm > 0.3:
                step = step * 0.3 / step_norm
            alpha = 1.0
            current_total = self.pes.energy(current_pos[0], current_pos[1]) + self._compute_bias_energy(current_pos[0], current_pos[1])
            for _ in range(10):
                new_pos = current_pos + alpha * step
                new_total = self.pes.energy(new_pos[0], new_pos[1]) + self._compute_bias_energy(new_pos[0], new_pos[1])
                if new_total < current_total:
                    break
                alpha *= 0.5
            current_pos = current_pos + alpha * step
        return current_pos

    def step(self):
        """执行一步DESW迭代"""
        # 交替前进和后退行走器
        is_forward = (self._iter_count % 2 == 0)

        if is_forward:
            current_pos = self.position_forward
            gaussians = self._gaussians_forward
            target = self.end_pos
        else:
            current_pos = self.position_backward
            gaussians = self._gaussians_backward
            target = self.start_pos

        # 1. 朝目标方向生成位移（加随机扰动）
        direction = target - current_pos
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            direction = direction / dir_norm
        else:
            direction = np.random.randn(2)
            direction = direction / np.linalg.norm(direction)

        direction = direction + np.random.randn(2) * 0.3
        direction = direction / np.linalg.norm(direction)

        # 2. 沿方向添加高斯偏置
        bias_center = current_pos + direction * self.step_size
        gaussians.append((bias_center.copy(), self.gaussian_height, self.gaussian_width))

        # 3. 在偏置势能面上局部优化
        new_pos = self._local_optimize_biased(current_pos)

        # 4. 在真实势能面上局部优化
        new_pos, new_energy = self._local_optimize(new_pos)

        # 5. Metropolis接受准则
        old_energy = float(self.pes.energy(current_pos[0], current_pos[1]))
        delta_E = new_energy - old_energy

        if delta_E < 0 or np.random.random() < np.exp(-delta_E / max(self.temperature, 1e-10)):
            if is_forward:
                self.position_forward = new_pos
                self._path_forward.append(new_pos.copy())
            else:
                self.position_backward = new_pos
                self._path_backward.append(new_pos.copy())

        # 更新位置
        self.position = self.position_forward.copy()

        # 检查两端路径是否相遇
        for pf in self._path_forward[-5:]:
            for pb in self._path_backward[-5:]:
                if np.linalg.norm(pf - pb) < self.step_size:
                    self._converged = True
                    break
            if self._converged:
                break

        # 更新最优
        pos_energy = float(self.pes.energy(self.position[0], self.position[1]))
        if pos_energy < self._best_energy:
            self._best_energy = pos_energy
            self._best_pos = self.position.copy()

        self._iter_count += 1

        state = {
            "position": self.position.copy(),
            "energy": pos_energy,
            "forward_pos": self.position_forward.copy(),
            "backward_pos": self.position_backward.copy(),
            "path_forward": [p.copy() for p in self._path_forward],
            "path_backward": [p.copy() for p in self._path_backward],
            "bias_gaussians": list(self._gaussians_forward) + list(self._gaussians_backward),
            "iteration": self._iter_count,
        }
        self._trajectory.append(state)
        return state

    def run(self) -> list[dict]:
        for _ in range(self.max_iter):
            if self._converged:
                break
            self.step()
        return self._trajectory

    def reset(self, start_pos: Optional[np.ndarray] = None, end_pos: Optional[np.ndarray] = None):
        if start_pos is not None:
            self.start_pos = np.array(start_pos, dtype=float)
        if end_pos is not None:
            self.end_pos = np.array(end_pos, dtype=float)

        self.position_forward = self.start_pos.copy()
        self.position_backward = self.end_pos.copy()
        self._gaussians_forward = []
        self._gaussians_backward = []
        self.position = self.start_pos.copy()
        self._start_pos = self.position.copy()
        self._converged = False
        self._trajectory = []
        self._iter_count = 0
        self._path_forward = [self.start_pos.copy()]
        self._path_backward = [self.end_pos.copy()]
        self._best_pos = self.start_pos.copy()
        self._best_energy = float(self.pes.energy(self.start_pos[0], self.start_pos[1]))

    def get_trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def is_converged(self) -> bool:
        return self._converged
