"""
势能面（Potential Energy Surface, PES）模块

提供多种经典势能面的定义，包括能量、梯度、Hessian矩阵的计算，
以及关键点（极小值、鞍点）的信息。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class PES(ABC):
    """势能面抽象基类，定义所有势能面必须实现的接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """势能面名称。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """势能面描述。"""
        ...

    @property
    @abstractmethod
    def bounds(self) -> Tuple[float, float, float, float]:
        """可视化范围，返回 (xmin, xmax, ymin, ymax)。"""
        ...

    @abstractmethod
    def energy(self, x: float, y: float) -> float:
        """计算给定坐标处的能量值。

        Parameters
        ----------
        x : float
            x 坐标
        y : float
            y 坐标

        Returns
        -------
        float
            能量值
        """
        ...

    @abstractmethod
    def gradient(self, x: float, y: float) -> np.ndarray:
        """计算给定坐标处的梯度向量（解析实现）。

        Parameters
        ----------
        x : float
            x 坐标
        y : float
            y 坐标

        Returns
        -------
        np.ndarray
            梯度向量 [dE/dx, dE/dy]，形状 (2,)
        """
        ...

    @abstractmethod
    def hessian(self, x: float, y: float) -> np.ndarray:
        """计算给定坐标处的 Hessian 矩阵（解析实现）。

        Parameters
        ----------
        x : float
            x 坐标
        y : float
            y 坐标

        Returns
        -------
        np.ndarray
            Hessian 矩阵，形状 (2, 2)
        """
        ...

    @abstractmethod
    def critical_points(self) -> List[Dict]:
        """返回已知关键点列表。

        Returns
        -------
        List[Dict]
            每个字典包含:
            - 'type': 'minimum' 或 'saddle'
            - 'pos': (x, y) 坐标元组
            - 'energy': 能量值
        """
        ...


class MuellerBrownPES(PES):
    """经典 Müller-Brown 势能面。

    E(x,y) = Σᵢ Aᵢ * exp(aᵢ(x-x₀ᵢ)² + bᵢ(x-x₀ᵢ)(y-y₀ᵢ) + cᵢ(y-y₀ᵢ)²)

    这是计算化学中广泛使用的基准势能面，具有三个极小值和两个鞍点。
    """

    # Müller-Brown 势能面参数
    _A = np.array([-200.0, -100.0, -170.0, 15.0])
    _a = np.array([-1.0, -1.0, -6.5, 0.7])
    _b = np.array([0.0, 0.0, 11.0, 0.6])
    _c = np.array([-10.0, -10.0, -6.5, 0.7])
    _x0 = np.array([1.0, 0.0, -0.5, -1.0])
    _y0 = np.array([0.0, 0.5, 1.5, 1.0])

    @property
    def name(self) -> str:
        return "Müller-Brown"

    @property
    def description(self) -> str:
        return "经典Müller-Brown势能面，具有三个极小值和两个鞍点"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-1.5, 1.2, -0.5, 2.0)

    def _exponent(self, x: float, y: float) -> np.ndarray:
        """计算每个高斯项的指数部分。"""
        dx = x - self._x0
        dy = y - self._y0
        return self._a * dx**2 + self._b * dx * dy + self._c * dy**2

    def _exp_terms(self, x: float, y: float) -> np.ndarray:
        """计算每个高斯项 Aᵢ * exp(...)。"""
        return self._A * np.exp(self._exponent(x, y))

    def energy(self, x: float, y: float) -> float:
        """计算 Müller-Brown 势能面的能量值。"""
        return float(np.sum(self._exp_terms(x, y)))

    def gradient(self, x: float, y: float) -> np.ndarray:
        """计算 Müller-Brown 势能面的解析梯度。

        dE/dx = Σᵢ Aᵢ * exp(...) * (2*aᵢ*(x-x₀ᵢ) + bᵢ*(y-y₀ᵢ))
        dE/dy = Σᵢ Aᵢ * exp(...) * (bᵢ*(x-x₀ᵢ) + 2*cᵢ*(y-y₀ᵢ))
        """
        dx = x - self._x0
        dy = y - self._y0
        exp_terms = self._exp_terms(x, y)

        dEdx = float(np.sum(exp_terms * (2.0 * self._a * dx + self._b * dy)))
        dEdy = float(np.sum(exp_terms * (self._b * dx + 2.0 * self._c * dy)))

        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        """计算 Müller-Brown 势能面的解析 Hessian 矩阵。

        d²E/dx² = Σᵢ Aᵢ*exp(...)*[(2aᵢ(x-x₀ᵢ)+bᵢ(y-y₀ᵢ))² + 2aᵢ]
        d²E/dy² = Σᵢ Aᵢ*exp(...)*[(bᵢ(x-x₀ᵢ)+2cᵢ(y-y₀ᵢ))² + 2cᵢ]
        d²E/dxdy = Σᵢ Aᵢ*exp(...)*[(2aᵢ(x-x₀ᵢ)+bᵢ(y-y₀ᵢ))*(bᵢ(x-x₀ᵢ)+2cᵢ(y-y₀ᵢ)) + bᵢ]
        """
        dx = x - self._x0
        dy = y - self._y0
        exp_terms = self._exp_terms(x, y)

        # 梯度分量因子
        fx = 2.0 * self._a * dx + self._b * dy  # dE/dx 中的括号部分
        fy = self._b * dx + 2.0 * self._c * dy  # dE/dy 中的括号部分

        d2Edx2 = float(np.sum(exp_terms * (fx**2 + 2.0 * self._a)))
        d2Edy2 = float(np.sum(exp_terms * (fy**2 + 2.0 * self._c)))
        d2Edxdy = float(np.sum(exp_terms * (fx * fy + self._b)))

        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        """返回 Müller-Brown 势能面的已知关键点。"""
        return [
            {
                "type": "minimum",
                "pos": (-0.558, 1.442),
                "energy": self.energy(-0.558, 1.442),
            },
            {
                "type": "minimum",
                "pos": (0.623, 0.028),
                "energy": self.energy(0.623, 0.028),
            },
            {
                "type": "saddle",
                "pos": (-0.822, 0.624),
                "energy": self.energy(-0.822, 0.624),
            },
        ]


class DoubleWellPES(PES):
    """双阱势能面。

    E(x,y) = (x²-1)² + y²

    在 x 方向具有两个对称的极小值，中间有一个鞍点。
    """

    @property
    def name(self) -> str:
        return "Double Well"

    @property
    def description(self) -> str:
        return "双阱势能面 E(x,y)=(x²-1)²+y²，具有两个对称极小值和一个鞍点"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-2.0, 2.0, -2.0, 2.0)

    def energy(self, x: float, y: float) -> float:
        """计算双阱势能面的能量值。"""
        return float((x**2 - 1) ** 2 + y**2)

    def gradient(self, x: float, y: float) -> np.ndarray:
        """计算双阱势能面的解析梯度。

        dE/dx = 4x(x²-1)
        dE/dy = 2y
        """
        dEdx = 4.0 * x * (x**2 - 1)
        dEdy = 2.0 * y
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        """计算双阱势能面的解析 Hessian 矩阵。

        d²E/dx² = 12x²-4
        d²E/dy² = 2
        d²E/dxdy = 0
        """
        d2Edx2 = 12.0 * x**2 - 4.0
        d2Edy2 = 2.0
        d2Edxdy = 0.0
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        """返回双阱势能面的已知关键点。"""
        return [
            {
                "type": "minimum",
                "pos": (-1.0, 0.0),
                "energy": self.energy(-1.0, 0.0),
            },
            {
                "type": "minimum",
                "pos": (1.0, 0.0),
                "energy": self.energy(1.0, 0.0),
            },
            {
                "type": "saddle",
                "pos": (0.0, 0.0),
                "energy": self.energy(0.0, 0.0),
            },
        ]


class ThreeMinimaPES(PES):
    """三极小值势能面。

    E(x,y) = (1/6)*x⁶ - (5/8)*x⁴ + (1/2)*x² + y²

    在 x 方向具有三个极小值（x≈-1.37, 0, 1.37）和两个鞍点。
    """

    @property
    def name(self) -> str:
        return "Three Minima"

    @property
    def description(self) -> str:
        return "三极小值势能面 E(x,y)=(1/6)x⁶-(5/8)x⁴+(1/2)x²+y²"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-2.5, 2.5, -2.0, 2.0)

    def energy(self, x: float, y: float) -> float:
        """计算三极小值势能面的能量值。"""
        return float((1.0 / 6.0) * x**6 - (5.0 / 8.0) * x**4 + (1.0 / 2.0) * x**2 + y**2)

    def gradient(self, x: float, y: float) -> np.ndarray:
        """计算三极小值势能面的解析梯度。

        dE/dx = x⁵ - 2.5x³ + x
        dE/dy = 2y
        """
        dEdx = x**5 - 2.5 * x**3 + x
        dEdy = 2.0 * y
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        """计算三极小值势能面的解析 Hessian 矩阵。

        d²E/dx² = 5x⁴ - 7.5x² + 1
        d²E/dy² = 2
        d²E/dxdy = 0
        """
        d2Edx2 = 5.0 * x**4 - 7.5 * x**2 + 1.0
        d2Edy2 = 2.0
        d2Edxdy = 0.0
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        """返回三极小值势能面的关键点（通过 scipy.optimize 数值求解）。"""
        try:
            from scipy.optimize import minimize
        except ImportError:
            # 如果 scipy 不可用，返回手动标注的近似值
            return self._approx_critical_points()

        critical_pts: List[Dict] = []
        found_positions: List[Tuple[float, float]] = []

        # 在多个起始点搜索极小值
        min_starts = [
            (-1.5, 0.0), (0.0, 0.0), (1.5, 0.0),
            (-1.0, 0.5), (1.0, -0.5), (0.0, 0.5),
        ]
        minima_positions: List[Tuple[float, float]] = []
        for x0 in min_starts:
            res = minimize(
                fun=lambda p: self.energy(p[0], p[1]),
                x0=x0,
                jac=lambda p: self.gradient(p[0], p[1]),
                method="L-BFGS-B",
                bounds=[(-2.5, 2.5), (-2.0, 2.0)],
            )
            if res.success:
                pos = (round(float(res.x[0]), 4), round(float(res.x[1]), 4))
                # 检查是否与已找到的点重复
                if not any(abs(pos[0] - fp[0]) < 0.05 and abs(pos[1] - fp[1]) < 0.05
                           for fp in found_positions):
                    found_positions.append(pos)
                    H = self.hessian(pos[0], pos[1])
                    eigenvalues = np.linalg.eigvalsh(H)
                    if np.all(eigenvalues > 0):
                        critical_pts.append({
                            "type": "minimum",
                            "pos": pos,
                            "energy": self.energy(pos[0], pos[1]),
                        })
                        minima_positions.append(pos)

        # 搜索鞍点：在极小值对之间的连线上搜索能量极大值
        for i in range(len(minima_positions)):
            for j in range(i + 1, len(minima_positions)):
                p1 = np.array(minima_positions[i])
                p2 = np.array(minima_positions[j])
                # 沿连线搜索能量极大值
                def line_energy(t: float, p1: np.ndarray = p1, p2: np.ndarray = p2) -> float:
                    p = p1 + t * (p2 - p1)
                    return self.energy(p[0], p[1])

                res = minimize(
                    fun=lambda t: -line_energy(t[0]),
                    x0=[0.5],
                    method="L-BFGS-B",
                    bounds=[(0.05, 0.95)],
                )
                if res.success:
                    t_opt = float(res.x[0])
                    p_saddle = p1 + t_opt * (p2 - p1)
                    pos = (round(float(p_saddle[0]), 4), round(float(p_saddle[1]), 4))
                    grad_norm = np.linalg.norm(self.gradient(pos[0], pos[1]))
                    if grad_norm < 0.5:
                        # 检查是否与已找到的点重复
                        if not any(abs(pos[0] - fp[0]) < 0.05 and abs(pos[1] - fp[1]) < 0.05
                                   for fp in found_positions):
                            found_positions.append(pos)
                            H = self.hessian(pos[0], pos[1])
                            eigenvalues = np.linalg.eigvalsh(H)
                            if np.any(eigenvalues < 0) and np.any(eigenvalues > 0):
                                critical_pts.append({
                                    "type": "saddle",
                                    "pos": pos,
                                    "energy": self.energy(pos[0], pos[1]),
                                })

        # 如果数值搜索未找到足够的关键点，返回近似值
        if len(critical_pts) < 3:
            return self._approx_critical_points()

        return critical_pts

    def _approx_critical_points(self) -> List[Dict]:
        """返回手动标注的近似关键点（当 scipy 不可用时的后备方案）。"""
        return [
            {
                "type": "minimum",
                "pos": (-1.3660, 0.0),
                "energy": self.energy(-1.3660, 0.0),
            },
            {
                "type": "minimum",
                "pos": (0.0, 0.0),
                "energy": self.energy(0.0, 0.0),
            },
            {
                "type": "minimum",
                "pos": (1.3660, 0.0),
                "energy": self.energy(1.3660, 0.0),
            },
            {
                "type": "saddle",
                "pos": (-0.7321, 0.0),
                "energy": self.energy(-0.7321, 0.0),
            },
            {
                "type": "saddle",
                "pos": (0.7321, 0.0),
                "energy": self.energy(0.7321, 0.0),
            },
        ]


class RosenbrockPES(PES):
    """Rosenbrock 势能面（香蕉函数）。

    E(x,y) = (1-x)² + 100(y-x²)²

    经典优化基准，具有一个狭长弯曲的谷底，全局极小值在(1,1)。
    """

    @property
    def name(self) -> str:
        return "Rosenbrock"

    @property
    def description(self) -> str:
        return "Rosenbrock函数 E=(1-x)²+100(y-x²)²，狭长弯曲谷底"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-2.0, 2.0, -1.0, 3.0)

    def energy(self, x: float, y: float) -> float:
        return float((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    def gradient(self, x: float, y: float) -> np.ndarray:
        dEdx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dEdy = 200 * (y - x ** 2)
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        d2Edx2 = 2 - 400 * y + 1200 * x ** 2
        d2Edy2 = 200.0
        d2Edxdy = -400 * x
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        return [
            {
                "type": "minimum",
                "pos": (1.0, 1.0),
                "energy": 0.0,
            },
        ]


class HimmelblauPES(PES):
    """Himmelblau 势能面。

    E(x,y) = (x²+y-11)² + (x+y²-7)²

    具有四个等价的极小值，是优化算法的经典测试函数。
    """

    @property
    def name(self) -> str:
        return "Himmelblau"

    @property
    def description(self) -> str:
        return "Himmelblau函数 E=(x²+y-11)²+(x+y²-7)²，四个等价极小值"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-5.0, 5.0, -5.0, 5.0)

    def energy(self, x: float, y: float) -> float:
        return float((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)

    def gradient(self, x: float, y: float) -> np.ndarray:
        dEdx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
        dEdy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        f1 = x ** 2 + y - 11
        f2 = x + y ** 2 - 7
        d2Edx2 = 12 * x ** 2 + 4 * y - 44 + 2
        d2Edy2 = 2 + 12 * y ** 2 + 4 * x - 28
        d2Edxdy = 4 * x + 4 * y
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        return [
            {"type": "minimum", "pos": (3.0, 2.0), "energy": 0.0},
            {"type": "minimum", "pos": (-2.8051, 3.1313), "energy": self.energy(-2.8051, 3.1313)},
            {"type": "minimum", "pos": (-3.7793, -3.2832), "energy": self.energy(-3.7793, -3.2832)},
            {"type": "minimum", "pos": (3.5844, -1.8481), "energy": self.energy(3.5844, -1.8481)},
        ]


class AckleyPES(PES):
    """Ackley 势能面（简化二维版）。

    E(x,y) = -20*exp(-0.2*sqrt(0.5*(x²+y²))) - exp(0.5*(cos(2πx)+cos(2πy))) + e + 20

    具有大量局部极小值，全局极小值在原点，是测试全局优化的经典函数。
    """

    @property
    def name(self) -> str:
        return "Ackley"

    @property
    def description(self) -> str:
        return "Ackley函数，具有大量局部极小值，全局极小值在原点"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-5.0, 5.0, -5.0, 5.0)

    def energy(self, x: float, y: float) -> float:
        r = np.sqrt(0.5 * (x ** 2 + y ** 2))
        cos_term = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
        return float(-20 * np.exp(-0.2 * r) - np.exp(cos_term) + np.e + 20)

    def gradient(self, x: float, y: float) -> np.ndarray:
        r = np.sqrt(0.5 * (x ** 2 + y ** 2))
        if r < 1e-10:
            r = 1e-10
        exp_r = np.exp(-0.2 * r)
        cos_term = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
        exp_cos = np.exp(cos_term)
        dEdx = 20 * 0.2 * x / (2 * r) * exp_r + np.pi * np.sin(2 * np.pi * x) * exp_cos
        dEdy = 20 * 0.2 * y / (2 * r) * exp_r + np.pi * np.sin(2 * np.pi * y) * exp_cos
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        # 使用数值差分计算Hessian（解析表达式过于复杂）
        h = 1e-5
        g0 = self.gradient(x, y)
        gx = self.gradient(x + h, y)
        gy = self.gradient(x, y + h)
        d2Edx2 = (gx[0] - g0[0]) / h
        d2Edy2 = (gy[1] - g0[1]) / h
        d2Edxdy = (gx[1] - g0[1]) / h
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        return [
            {"type": "minimum", "pos": (0.0, 0.0), "energy": 0.0},
        ]


class RastriginPES(PES):
    """Rastrigin 势能面。

    E(x,y) = 20 + x² - 10*cos(2πx) + y² - 10*cos(2πy)

    具有大量规则分布的局部极小值，全局极小值在原点。
    """

    @property
    def name(self) -> str:
        return "Rastrigin"

    @property
    def description(self) -> str:
        return "Rastrigin函数，大量规则分布的局部极小值，全局极小值在原点"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (-5.12, 5.12, -5.12, 5.12)

    def energy(self, x: float, y: float) -> float:
        return float(20 + x ** 2 - 10 * np.cos(2 * np.pi * x)
                     + y ** 2 - 10 * np.cos(2 * np.pi * y))

    def gradient(self, x: float, y: float) -> np.ndarray:
        dEdx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        dEdy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        d2Edx2 = 2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x)
        d2Edy2 = 2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * y)
        d2Edxdy = 0.0
        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        return [
            {"type": "minimum", "pos": (0.0, 0.0), "energy": 0.0},
        ]


class CustomPES(PES):
    """自定义势能面。

    接受用户输入的函数表达式字符串，使用 eval 计算能量，
    梯度和 Hessian 用数值差分实现。
    """

    def __init__(
        self,
        expression: str,
        bounds: Tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
        pes_name: str = "Custom",
        pes_description: str = "自定义势能面",
    ) -> None:
        """初始化自定义势能面。

        Parameters
        ----------
        expression : str
            能量函数表达式字符串，变量为 x 和 y。
            例如: "(x**2 - 1)**2 + y**2"
        bounds : Tuple[float, float, float, float], optional
            可视化范围 (xmin, xmax, ymin, ymax)，默认 (-3, 3, -3, 3)
        pes_name : str, optional
            势能面名称，默认 "Custom"
        pes_description : str, optional
            势能面描述，默认 "自定义势能面"
        """
        self._expression = expression
        # 预处理：将常见书面公式写法转换为Python表达式
        import re
        expr = self._expression

        # 1. ^ 转换为 ** （幂运算）
        expr = re.sub(r'\^', '**', expr)

        # 2. 添加隐式乘法
        expr = re.sub(r'(\d)([xy])', r'\1*\2', expr)      # 2y → 2*y, 3x → 3*x
        expr = re.sub(r'(\d)\(', r'\1*(', expr)             # 2( → 2*(
        expr = re.sub(r'\)(\d)', r')*\1', expr)             # )2 → )*2
        expr = re.sub(r'\)\(', r')*(', expr)                # )( → )*(
        expr = re.sub(r'\)([xy])', r')*\1', expr)           # )x → )*x, )y → )*y
        expr = re.sub(r'([xy])\(', r'\1*(', expr)           # x( → x*(
        expr = re.sub(r'([xy])(\d)', r'\1*\2', expr)        # x2 → x*2 (罕见但可能)

        # 3. 数字与函数名之间的隐式乘法：2sin → 2*sin
        func_names = r'(?:sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|arcsin|arccos|arctan|arcsinh|arccosh|arctanh|exp|log|ln|log2|log10|sqrt|cbrt|abs|sec|csc|cot)'
        expr = re.sub(r'(\d)(' + func_names + r')', r'\1*\2', expr)  # 2sin → 2*sin
        expr = re.sub(r'\)(' + func_names + r')', r')*\1', expr)      # )sin → )*sin

        # 4. 变量与函数名之间的隐式乘法：x sin → x*sin（罕见）
        # 注意：不处理 x sin(y) → x*sin(y)，因为空格可能是分隔符

        self._expression = expr
        self._bounds = bounds
        self._pes_name = pes_name
        self._pes_description = pes_description
        # 数值差分步长
        self._h = 1e-5

    @property
    def name(self) -> str:
        return self._pes_name

    @property
    def description(self) -> str:
        return self._pes_description

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._bounds

    # 支持的数学函数命名空间
    _MATH_NAMESPACE = {}
    _MATH_NAMESPACE.update({
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sqrt': np.sqrt, 'cbrt': np.cbrt,
        'exp': np.exp, 'exp2': np.exp2, 'expm1': np.expm1,
        'log': np.log, 'ln': np.log, 'log2': np.log2, 'log10': np.log10, 'log1p': np.log1p,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan, 'atan2': np.arctan2,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'arcsinh': np.arcsinh, 'arccosh': np.arccosh, 'arctanh': np.arctanh,
        'pi': np.pi, 'e': np.e,
        'sign': np.sign, 'floor': np.floor, 'ceil': np.ceil,
        'pow': np.power, 'power': np.power,
        'hypot': np.hypot,
        'sec': lambda x: 1.0 / np.cos(x), 'csc': lambda x: 1.0 / np.sin(x), 'cot': lambda x: np.cos(x) / np.sin(x),
    })

    def energy(self, x: float, y: float) -> float:
        """使用 eval 计算自定义表达式的能量值。支持常见数学函数。"""
        ns = {'x': x, 'y': y}
        ns.update(self._MATH_NAMESPACE)
        return float(eval(self._expression, {"__builtins__": {}}, ns))

    def gradient(self, x: float, y: float) -> np.ndarray:
        """使用中心差分计算数值梯度。

        dE/dx ≈ (E(x+h, y) - E(x-h, y)) / (2h)
        dE/dy ≈ (E(x, y+h) - E(x, y-h)) / (2h)
        """
        h = self._h
        dEdx = (self.energy(x + h, y) - self.energy(x - h, y)) / (2.0 * h)
        dEdy = (self.energy(x, y + h) - self.energy(x, y - h)) / (2.0 * h)
        return np.array([dEdx, dEdy])

    def hessian(self, x: float, y: float) -> np.ndarray:
        """使用中心差分计算数值 Hessian 矩阵。

        d²E/dx² ≈ (E(x+h,y) - 2E(x,y) + E(x-h,y)) / h²
        d²E/dy² ≈ (E(x,y+h) - 2E(x,y) + E(x,y-h)) / h²
        d²E/dxdy ≈ (E(x+h,y+h) - E(x+h,y-h) - E(x-h,y+h) + E(x-h,y-h)) / (4h²)
        """
        h = self._h
        E0 = self.energy(x, y)

        d2Edx2 = (self.energy(x + h, y) - 2.0 * E0 + self.energy(x - h, y)) / (h**2)
        d2Edy2 = (self.energy(x, y + h) - 2.0 * E0 + self.energy(x, y - h)) / (h**2)
        d2Edxdy = (
            self.energy(x + h, y + h)
            - self.energy(x + h, y - h)
            - self.energy(x - h, y + h)
            + self.energy(x - h, y - h)
        ) / (4.0 * h**2)

        return np.array([[d2Edx2, d2Edxdy],
                         [d2Edxdy, d2Edy2]])

    def critical_points(self) -> List[Dict]:
        """自定义势能面无预知关键点，返回空列表。"""
        return []


def get_pes(name: str) -> PES:
    """工厂函数，根据名称返回势能面实例。

    Parameters
    ----------
    name : str
        势能面名称，可选值:
        - 'mueller_brown': Müller-Brown 势能面
        - 'double_well': 双阱势能面
        - 'three_minima': 三极小值势能面

    Returns
    -------
    PES
        对应的势能面实例

    Raises
    ------
    ValueError
        当 name 不是已知势能面名称时抛出
    """
    pes_map = {
        "mueller_brown": MuellerBrownPES,
        "double_well": DoubleWellPES,
        "three_minima": ThreeMinimaPES,
        "rosenbrock": RosenbrockPES,
        "himmelblau": HimmelblauPES,
        "ackley": AckleyPES,
        "rastrigin": RastriginPES,
    }

    if name not in pes_map:
        available = ", ".join(pes_map.keys())
        raise ValueError(f"未知势能面名称 '{name}'，可选值: {available}")

    return pes_map[name]()
