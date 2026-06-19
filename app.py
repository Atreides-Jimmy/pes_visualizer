"""
势能面搜索算法交互式可视化教学工具

基于 PyQt5 + PyQtGraph 实现的 GUI 应用，
支持牛顿法、Dimer方法、NEB/CI-NEB方法的交互式可视化。
"""

import sys
import os
import csv
import io

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QCheckBox, QRadioButton, QButtonGroup,
    QLineEdit, QTextBrowser, QSplitter,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QInputDialog, QStackedWidget, QSlider,
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QImage, QVector3D, QIcon

import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotCurveItem, ScatterPlotItem, ImageItem, ColorBarItem
from pyqtgraph import TextItem

# 尝试导入OpenGL模块（3D视图需要）
try:
    from pyqtgraph.opengl import GLViewWidget, GLMeshItem, GLScatterPlotItem, GLLinePlotItem, GLAxisItem, GLSurfacePlotItem, GLTextItem
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

# 设置pyqtgraph配置
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1a1b2e')
pg.setConfigOption('foreground', '#c8cce0')

# 尝试导入skimage用于等高线计算
try:
    from skimage.measure import find_contours
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from pes import MuellerBrownPES, DoubleWellPES, ThreeMinimaPES, RosenbrockPES, HimmelblauPES, AckleyPES, RastriginPES, CustomPES, get_pes
from algorithms import NewtonMethod, DimerMethod, NEBMethod, SteepestDescentMethod, BasinHoppingMethod, MetropolisMCMethod, SSWMethod, MetaDynamicsMethod, MinimaHoppingMethod, GeneticAlgorithmMethod, PSOMethod, ABCMethod, UmbrellaSamplingMethod, ABFMethod, CBDMethod, DESWMethod


# ======================== 全局QSS样式（现代科技暗色主题） ========================

GLOBAL_STYLESHEET = """
    * {
        font-family: "Segoe UI", "微软雅黑", sans-serif;
        font-size: 9pt;
    }
    QMainWindow {
        background-color: #1a1b2e;
    }
    QWidget {
        color: #e0e0e0;
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #2d3050;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 14px;
        background-color: #22243a;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #00d4ff;
        font-size: 9pt;
    }
    QGroupBox[flat="true"] {
        border: none;
        background-color: transparent;
        margin-top: 12px;
        padding-top: 12px;
    }
    QGroupBox[flat="true"]::title {
        color: #00d4ff;
        font-size: 9pt;
    }
    QGroupBox#infoPhysics, QGroupBox#infoPrinciple, QGroupBox#infoTerms, QGroupBox#infoResult {
        margin-top: 14px;
        padding-top: 14px;
        border: 1px solid #2d3050;
        background-color: #1e2035;
    }
    QGroupBox#infoPhysics::title, QGroupBox#infoPrinciple::title, QGroupBox#infoTerms::title, QGroupBox#infoResult::title {
        subcontrol-origin: margin;
        left: 6px;
        padding: 0 4px;
        color: #00d4ff;
    }
    QPushButton {
        border: 1px solid #3a3d5c;
        border-radius: 4px;
        padding: 5px 12px;
        background-color: #2a2d48;
        color: #c8cce0;
        min-height: 22px;
    }
    QPushButton:hover {
        background-color: #363a5c;
        border-color: #00d4ff;
        color: #ffffff;
    }
    QPushButton:pressed {
        background-color: #1a1d30;
    }
    QPushButton#playBtn {
        background-color: #00b894;
        color: #ffffff;
        border: 1px solid #00a381;
        font-weight: bold;
    }
    QPushButton#playBtn:hover {
        background-color: #00d9a6;
        border-color: #00d4ff;
    }
    QPushButton#pauseBtn {
        background-color: #e17055;
        color: #ffffff;
        border: 1px solid #c0604a;
        font-weight: bold;
    }
    QPushButton#pauseBtn:hover {
        background-color: #f0876e;
        border-color: #00d4ff;
    }
    QPushButton#resetBtn {
        background-color: #636e72;
        color: #ffffff;
        border: 1px solid #535c60;
        font-weight: bold;
    }
    QPushButton#resetBtn:hover {
        background-color: #7c8a8e;
        border-color: #00d4ff;
    }
    QPushButton#setPointBtn {
        background-color: #0984e3;
        color: #ffffff;
        border: 1px solid #0872c4;
        font-weight: bold;
    }
    QPushButton#setPointBtn:hover {
        background-color: #2d9bf0;
        border-color: #00d4ff;
    }
    QPushButton#finishPointBtn {
        background-color: #00b894;
        color: #ffffff;
        border: 1px solid #00a381;
        font-weight: bold;
    }
    QPushButton#finishPointBtn:hover {
        background-color: #00d9a6;
        border-color: #00d4ff;
    }
    QPushButton#customConfirmBtn {
        background-color: #e17055;
        color: #ffffff;
        border: 1px solid #c0604a;
        font-weight: bold;
        padding: 2px 8px;
    }
    QPushButton#customConfirmBtn:hover {
        background-color: #f0876e;
        border-color: #00d4ff;
    }
    QPushButton#applyRangeBtn {
        background-color: #0984e3;
        color: #ffffff;
        border: 1px solid #0872c4;
        font-weight: bold;
        padding: 2px 8px;
    }
    QPushButton#applyRangeBtn:hover {
        background-color: #2d9bf0;
        border-color: #00d4ff;
    }
    QComboBox {
        border: 1px solid #3a3d5c;
        border-radius: 4px;
        padding: 3px 8px;
        background-color: #2a2d48;
        color: #e0e0e0;
        min-height: 20px;
    }
    QComboBox:hover {
        border-color: #00d4ff;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #00d4ff;
    }
    QComboBox QAbstractItemView {
        border: 1px solid #3a3d5c;
        background-color: #2a2d48;
        color: #e0e0e0;
        selection-background-color: #0984e3;
        selection-color: #ffffff;
    }
    QDoubleSpinBox, QSpinBox {
        border: 1px solid #3a3d5c;
        border-radius: 4px;
        padding: 3px 6px;
        background-color: #2a2d48;
        color: #e0e0e0;
        min-height: 20px;
    }
    QDoubleSpinBox:hover, QSpinBox:hover {
        border-color: #00d4ff;
    }
    QDoubleSpinBox::up-button, QSpinBox::up-button {
        background-color: #363a5c;
        border: none;
        width: 16px;
    }
    QDoubleSpinBox::down-button, QSpinBox::down-button {
        background-color: #363a5c;
        border: none;
        width: 16px;
    }
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid #00d4ff;
    }
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid #00d4ff;
    }
    QLineEdit {
        border: 1px solid #3a3d5c;
        border-radius: 4px;
        padding: 3px 6px;
        background-color: #2a2d48;
        color: #e0e0e0;
        min-height: 20px;
    }
    QLineEdit:hover {
        border-color: #00d4ff;
    }
    QCheckBox {
        spacing: 6px;
        color: #e0e0e0;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #3a3d5c;
        border-radius: 3px;
        background-color: #2a2d48;
    }
    QCheckBox::indicator:checked {
        background-color: #0984e3;
        border-color: #0984e3;
    }
    QRadioButton {
        spacing: 6px;
        color: #e0e0e0;
    }
    QRadioButton::indicator {
        width: 12px;
        height: 12px;
        border: 1px solid #3a3d5c;
        border-radius: 6px;
        background-color: #2a2d48;
    }
    QRadioButton::indicator:checked {
        background-color: #0984e3;
        border-color: #0984e3;
    }
    QTextBrowser {
        border: 1px solid #2d3050;
        border-radius: 4px;
        background-color: #1e2035;
        color: #c8cce0;
    }
    QTableWidget {
        border: 1px solid #2d3050;
        gridline-color: #2d3050;
        background-color: #1e2035;
        color: #e0e0e0;
    }
    QHeaderView::section {
        background-color: #2a2d48;
        border: 1px solid #2d3050;
        padding: 3px 6px;
        font-weight: bold;
        color: #00d4ff;
    }
    QSplitter::handle {
        background-color: #2d3050;
    }
    QSplitter::handle:horizontal {
        width: 3px;
        background-color: #2d3050;
    }
    QSplitter::handle:horizontal:hover {
        background-color: #00d4ff;
        width: 5px;
    }
    QSplitter::handle:vertical {
        height: 3px;
        background-color: #2d3050;
    }
    QSplitter::handle:vertical:hover {
        background-color: #00d4ff;
        height: 5px;
    }
    QScrollBar:vertical {
        background-color: #1a1b2e;
        width: 8px;
        border: none;
    }
    QScrollBar::handle:vertical {
        background-color: #3a3d5c;
        border-radius: 4px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #00d4ff;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        background-color: #1a1b2e;
        height: 8px;
        border: none;
    }
    QScrollBar::handle:horizontal {
        background-color: #3a3d5c;
        border-radius: 4px;
        min-width: 30px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #00d4ff;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    QToolTip {
        background-color: #2a2d48;
        color: #e0e0e0;
        border: 1px solid #00d4ff;
        border-radius: 4px;
        padding: 4px;
    }
    QLabel {
        color: #c8cce0;
    }
"""


# ======================== 势能面可视化画布 ========================

class PESCanvas(QWidget):
    """势能面可视化画布，支持等高线和3D模式

    使用PyQtGraph替代Matplotlib实现：
    - 2D等高线：pg.PlotWidget + pg.ImageItem + pg.ScatterPlotItem + pg.PlotCurveItem
    - 3D曲面：pg.opengl.GLViewWidget + GLMeshItem
    - 能量曲线：pg.PlotWidget
    - 使用QStackedWidget在2D/3D视图间切换
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.pes = None
        self.X = None
        self.Y = None
        self.Z = None
        self.mode = 'contour'  # 'contour' 或 '3d'
        self._click_callback = None

        # 用户视图状态
        self._user_xlim = None
        self._user_ylim = None
        self._user_dist = None  # 3D缩放距离（兼容属性）

        self._custom_bounds = None
        self._z_scale = 1.0
        self._z_offset = 0.0
        self._original_bounds = None
        self._bias_function = None  # 偏置势函数（用于SSW/MetaDynamics可视化）
        self._Z_original = None  # 无偏置的原始Z数据
        self._reset_camera_center = False  # 是否重置3D相机中心到新PES中心
        self._legend_item = None  # 等高线图图例项（确保唯一）

        # 当前布局状态
        self._current_layout = None  # 'contour', 'contour+energy', '3d', '3d+energy'

        # 兼容属性：_main_ax 提供 set_xlim/set_ylim 方法
        self._main_ax = _ViewRangeProxy(self)
        self._cbar_ax = None  # 兼容属性，不再使用

        # 重绘回调
        self._redraw_callback = None

        # 构建UI
        self._init_ui()

    def _init_ui(self):
        """初始化UI组件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 使用QStackedWidget在2D/3D视图间切换
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # ---- 2D等高线页面 ----
        self._contour_page = QWidget()
        contour_layout = QVBoxLayout(self._contour_page)
        contour_layout.setContentsMargins(0, 0, 0, 0)
        contour_layout.setSpacing(0)

        # 主等高线图 + colorbar
        self._contour_splitter = QSplitter(Qt.Horizontal)
        self._contour_widget = PlotWidget()
        self._contour_widget.setAspectLocked(False)
        self._contour_widget.setLabel('bottom', 'x')
        self._contour_widget.setLabel('left', 'y')
        self._contour_widget.setTitle('势能面等高线图')
        self._contour_splitter.addWidget(self._contour_widget)

        # ColorBar widget
        self._colorbar_widget = pg.HistogramLUTWidget()
        self._colorbar_widget.setMaximumWidth(120)
        self._colorbar_widget.setMinimumWidth(80)
        self._contour_splitter.addWidget(self._colorbar_widget)
        self._contour_splitter.setSizes([800, 100])

        contour_layout.addWidget(self._contour_splitter)

        # 能量曲线子图（初始隐藏）
        self._energy_contour_widget = PlotWidget()
        self._energy_contour_widget.setMaximumHeight(200)
        self._energy_contour_widget.setLabel('bottom', '路径距离')
        self._energy_contour_widget.setLabel('left', '能量 E')
        self._energy_contour_widget.setTitle('沿路径能量曲线')
        self._energy_contour_widget.hide()
        contour_layout.addWidget(self._energy_contour_widget)

        self._stack.addWidget(self._contour_page)

        # ---- 3D曲面页面 ----
        self._3d_page = QWidget()
        layout_3d = QVBoxLayout(self._3d_page)
        layout_3d.setContentsMargins(0, 0, 0, 0)
        layout_3d.setSpacing(0)

        if HAS_OPENGL:
            self._gl_widget = GLViewWidget()
            self._gl_widget.setBackgroundColor('#1a1b2e')
            self._gl_widget.setCameraPosition(distance=20, elevation=30, azimuth=45)
            layout_3d.addWidget(self._gl_widget)
            # 平移偏移（相机空间，世界单位），用于右键拖拽平移
            # 通过覆盖viewMatrix实现，保持opts['center']不变，旋转中心始终在PES中心
            self._pan_offset = np.array([0.0, 0.0, 0.0])
            self._original_view_matrix = self._gl_widget.viewMatrix
            self._gl_widget.viewMatrix = self._pan_view_matrix
            # 3D视图右上角半透明图例标签
            self._3d_legend_label = QLabel(self._gl_widget)
            self._3d_legend_label.setStyleSheet(
                "background-color: rgba(255, 255, 255, 180);"
                "color: #222;"
                "border: 1px solid rgba(100, 100, 100, 200);"
                "padding: 4px 6px;"
                "font-size: 11px;"
            )
            self._3d_legend_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self._3d_legend_label.setText("")
            self._3d_legend_label.setVisible(False)
        else:
            # 没有OpenGL时显示提示
            label = QLabel("3D视图需要PyOpenGL支持，请安装: pip install PyOpenGL")
            label.setAlignment(Qt.AlignCenter)
            layout_3d.addWidget(label)
            self._gl_widget = None
            self._3d_legend_label = None

        # 3D能量曲线子图（初始隐藏）
        self._energy_3d_widget = PlotWidget()
        self._energy_3d_widget.setMaximumHeight(200)
        self._energy_3d_widget.setLabel('bottom', '路径距离')
        self._energy_3d_widget.setLabel('left', '能量 E')
        self._energy_3d_widget.setTitle('沿路径能量曲线')
        self._energy_3d_widget.hide()
        layout_3d.addWidget(self._energy_3d_widget)

        self._stack.addWidget(self._3d_page)

        # 默认显示等高线
        self._stack.setCurrentIndex(0)

        # 连接点击事件
        self._contour_widget.scene().sigMouseClicked.connect(self._on_contour_click)
        # GLViewWidget没有sigMouseClicked信号，使用mousePressEvent/mouseReleaseEvent重写方式
        if self._gl_widget is not None:
            self._gl_widget.mousePressEvent = self._make_3d_click_handler(self._gl_widget.mousePressEvent)
            self._gl_widget.mouseReleaseEvent = self._on_3d_mouse_release
            # 右键拖拽平移势能面（连同坐标轴）
            self._gl_widget.mouseMoveEvent = self._make_3d_move_handler(self._gl_widget.mouseMoveEvent)

        # 存储当前绘图项引用，用于清除和重建
        self._contour_items = []
        self._3d_items = []
        self._image_item = None
        self._contour_line_items = []

    def set_click_callback(self, callback):
        """设置鼠标点击回调函数"""
        self._click_callback = callback

    def _make_3d_click_handler(self, original_handler):
        """创建3D视图的鼠标点击处理器

        GLViewWidget没有sigMouseClicked信号，需要通过包装mousePressEvent来拦截点击。
        只在左键单击（非拖拽）时触发选点回调。
        """
        def handler(ev):
            # 先调用原始处理器（旋转等）
            original_handler(ev)
            # 只处理左键
            if ev.button() != Qt.LeftButton:
                return
            # 记录按下位置，在release时判断是否为单击
            self._3d_press_pos = ev.pos()
        return handler

    def _pan_view_matrix(self):
        """覆盖GLViewWidget的viewMatrix，在相机空间应用平移偏移

        保持opts['center']不变（旋转中心始终在PES中心），
        平移在相机空间应用（在距离平移之后、旋转之前），
        这样平移方向始终与屏幕/鼠标方向一致，与势能面旋转角度无关。
        """
        from PyQt5.QtGui import QMatrix4x4
        tr = QMatrix4x4()
        gl = self._gl_widget
        # 1. 距离平移（相机后退）
        tr.translate(0.0, 0.0, -gl.opts['distance'])
        # 2. 相机空间平移偏移（屏幕空间：+X右，+Y上）
        #    在旋转之前应用，确保平移方向与屏幕一致，不受旋转影响
        #    鼠标右移(dx>0)→pan_offset[0]增大→场景右移→translate(+pan_x)
        #    鼠标上移(dy>0)→pan_offset[1]增大→场景上移→translate(-pan_y)（Y轴翻转）
        tr.translate(self._pan_offset[0], -self._pan_offset[1], 0)
        # 3. 旋转
        if gl.opts['rotationMethod'] == 'quaternion':
            tr.rotate(gl.opts['rotation'])
        else:
            tr.rotate(gl.opts['elevation']-90, 1, 0, 0)
            tr.rotate(gl.opts['azimuth']+90, 0, 0, -1)
        # 4. 平移到中心（旋转中心，保持PES中心为旋转锚点）
        center = gl.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr

    def _make_3d_move_handler(self, original_move_handler):
        """创建3D视图的鼠标移动处理器，支持右键拖拽平移势能面（连同坐标轴）

        右键按住后移动：通过viewMatrix偏移平移视图，旋转中心始终在PES中心
        平移方向始终与鼠标移动方向一致，与势能面旋转角度无关
        其他情况：调用原始处理器（左键旋转等）
        """
        def handler(ev):
            if ev.buttons() & Qt.RightButton:
                # 右键拖拽：平移
                lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
                if not hasattr(self._gl_widget, 'mousePos'):
                    self._gl_widget.mousePos = lpos
                diff = lpos - self._gl_widget.mousePos
                self._gl_widget.mousePos = lpos
                # 计算像素到世界单位的转换因子（与pyqtgraph的pan一致）
                gl = self._gl_widget
                from math import tan, radians
                dist = (gl.opts['center'] - gl.cameraPosition()).length()
                fov_factor = tan(radians(gl.opts['fov']) / 2) * 2
                scale_factor = dist * fov_factor / gl.width()
                # 累积平移偏移（相机空间：+X右，+Y上；屏幕：+X右，+Y下）
                # 鼠标右移(dx>0)→场景右移→相机左移→viewMatrix translate +X→pan_offset_x += scale*dx
                self._pan_offset[0] += scale_factor * diff.x()
                self._pan_offset[1] += scale_factor * diff.y()
                gl.update()
            else:
                # 调用原始处理器（左键旋转等）
                original_move_handler(ev)
        return handler

    def _on_3d_mouse_release(self, ev):
        """3D视图鼠标释放事件 - 判断是否为单击并触发选点"""
        if self._click_callback is None:
            return
        if self.mode != '3d':
            return
        if not HAS_OPENGL or self._gl_widget is None:
            return
        if ev.button() != Qt.LeftButton:
            return
        # 判断是否为单击（移动距离小于5像素）
        press_pos = getattr(self, '_3d_press_pos', None)
        if press_pos is not None:
            delta = ev.pos() - press_pos
            if abs(delta.x()) < 5 and abs(delta.y()) < 5:
                self._do_3d_pick(ev.pos().x(), ev.pos().y())

    def _do_3d_pick(self, mx, my):
        """执行3D选点：将网格点投影到屏幕，找最近的

        注意：3D渲染时Z轴数据被缩放，所以投影匹配也需要使用缩放后的Z值
        """
        if self.X is None or self.Y is None or self.Z is None:
            return
        if self.pes is None:
            return
        # 获取Z轴缩放参数
        z_scale = getattr(self, '_z_scale', 1.0)
        z_offset = getattr(self, '_z_offset', 0.0)
        try:
            from PyQt5.QtGui import QMatrix4x4, QVector4D
            view = self._gl_widget.viewMatrix()
            proj = self._gl_widget.projectionMatrix()
            w = self._gl_widget.width()
            h = self._gl_widget.height()

            # 降采样以提升性能（使用更细的步长以覆盖更多区域）
            step = max(1, self.X.shape[0] // 100)
            xs = self.X[::step, ::step].ravel()
            ys = self.Y[::step, ::step].ravel()
            zs = self.Z[::step, ::step].ravel()
            # 使用缩放后的Z值进行投影匹配（与渲染一致）
            zs_scaled = (zs - z_offset) * z_scale

            best_x, best_y = None, None
            best_dist = float('inf')

            for i in range(len(xs)):
                if not np.isfinite(zs[i]):
                    continue
                # QMatrix4x4.map()需要QVector4D，不接受numpy数组
                p = QVector4D(float(xs[i]), float(ys[i]), float(zs_scaled[i]), 1.0)
                p_view = view.map(p)
                p_proj = proj.map(p_view)
                w_val = p_proj.w()
                if w_val == 0:
                    continue
                ndc_x = p_proj.x() / w_val
                ndc_y = p_proj.y() / w_val
                sx = (ndc_x + 1) * 0.5 * w
                sy = (1 - ndc_y) * 0.5 * h
                dist = (sx - mx)**2 + (sy - my)**2
                if dist < best_dist:
                    best_dist = dist
                    best_x = xs[i]
                    best_y = ys[i]

            if best_x is not None and best_dist < 100000:
                # scipy优化精确定位
                try:
                    from scipy.optimize import minimize as scipy_minimize

                    def screen_distance(params):
                        px, py = float(params[0]), float(params[1])
                        pz = self.pes.energy(px, py)
                        if not np.isfinite(pz):
                            return 1e10
                        pz_s = (pz - z_offset) * z_scale
                        p = QVector4D(px, py, pz_s, 1.0)
                        p_view = view.map(p)
                        p_proj = proj.map(p_view)
                        w_val = p_proj.w()
                        if w_val == 0:
                            return 1e10
                        ndc_x = p_proj.x() / w_val
                        ndc_y = p_proj.y() / w_val
                        sx = (ndc_x + 1) * 0.5 * w
                        sy = (1 - ndc_y) * 0.5 * h
                        return (sx - mx)**2 + (sy - my)**2

                    result = scipy_minimize(
                        screen_distance,
                        x0=[best_x, best_y],
                        method='Nelder-Mead',
                        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200}
                    )
                    if result.success and np.isfinite(result.x[0]) and np.isfinite(result.x[1]):
                        opt_x, opt_y = result.x
                        opt_dist = screen_distance([opt_x, opt_y])
                        grid_dist = screen_distance([best_x, best_y])
                        if opt_dist <= grid_dist * 1.5:
                            best_x = opt_x
                            best_y = opt_y
                except Exception:
                    pass

                # 限制在网格数据范围内（比pes.bounds更大）
                if self.X is not None and self.Y is not None:
                    best_x = np.clip(best_x, self.X.min(), self.X.max())
                    best_y = np.clip(best_y, self.Y.min(), self.Y.max())
                self._click_callback(float(best_x), float(best_y))
        except Exception:
            pass

    def _on_contour_click(self, event):
        """处理2D等高线模式下的鼠标点击"""
        if self._click_callback is None:
            return
        if self.mode != 'contour':
            return
        # 只处理左键点击
        if event.button() != Qt.LeftButton:
            return

        try:
            vb = self._contour_widget.plotItem.vb
            pos = event.scenePos()
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if np.isfinite(x) and np.isfinite(y):
                # 限制在网格数据范围内（比pes.bounds更大，因为compute_grid扩展了1.5倍）
                if self.X is not None and self.Y is not None:
                    x = np.clip(x, self.X.min(), self.X.max())
                    y = np.clip(y, self.Y.min(), self.Y.max())
                self._click_callback(x, y)
        except Exception:
            pass

    def _maybe_expand_grid(self, new_xlim, new_ylim):
        """检查视图范围是否超出已计算的网格，若超出则动态扩展。"""
        if self.pes is None or self.X is None:
            return
        grid_xmin, grid_xmax = self.X.min(), self.X.max()
        grid_ymin, grid_ymax = self.Y.min(), self.Y.max()
        need_expand = (new_xlim[0] < grid_xmin or new_xlim[1] > grid_xmax or
                       new_ylim[0] < grid_ymin or new_ylim[1] > grid_ymax)
        if not need_expand:
            return
        xcenter = (new_xlim[0] + new_xlim[1]) / 2
        ycenter = (new_ylim[0] + new_ylim[1]) / 2
        xhalf = max((new_xlim[1] - new_xlim[0]), (grid_xmax - grid_xmin)) * 1.2
        yhalf = max((new_ylim[1] - new_ylim[0]), (grid_ymax - grid_ymin)) * 1.2
        resolution = 200
        x = np.linspace(xcenter - xhalf, xcenter + xhalf, resolution)
        y = np.linspace(ycenter - yhalf, ycenter + yhalf, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        try:
            self.Z = np.vectorize(self.pes.energy)(self.X, self.Y)
        except Exception:
            self.Z = np.zeros_like(self.X)
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        self.Z[i, j] = self.pes.energy(self.X[i, j], self.Y[i, j])
                    except Exception:
                        self.Z[i, j] = np.nan

    def keyPressEvent(self, event):
        """处理键盘方向键平移等高线图"""
        if self.mode != 'contour':
            super().keyPressEvent(event)
            return

        pan_step = 0.1
        vb = self._contour_widget.plotItem.vb
        view_range = vb.viewRange()
        xlim = view_range[0]
        ylim = view_range[1]
        dx = (xlim[1] - xlim[0]) * pan_step
        dy = (ylim[1] - ylim[0]) * pan_step

        key = event.key()
        if key == Qt.Key_Left:
            new_xlim = [xlim[0] - dx, xlim[1] - dx]
            new_ylim = list(ylim)
        elif key == Qt.Key_Right:
            new_xlim = [xlim[0] + dx, xlim[1] + dx]
            new_ylim = list(ylim)
        elif key == Qt.Key_Up:
            new_xlim = list(xlim)
            new_ylim = [ylim[0] + dy, ylim[1] + dy]
        elif key == Qt.Key_Down:
            new_xlim = list(xlim)
            new_ylim = [ylim[0] - dy, ylim[1] - dy]
        else:
            super().keyPressEvent(event)
            return

        # 限制平移范围
        if self.pes is not None:
            xmin, xmax, ymin, ymax = self.pes.bounds
            range_x = (xmax - xmin)
            range_y = (ymax - ymin)
            max_offset = 25
            if abs(new_xlim[0] - xmin) > range_x * max_offset:
                return
            if abs(new_ylim[0] - ymin) > range_y * max_offset:
                return

        vb.setRange(xRange=new_xlim, yRange=new_ylim, padding=0)
        self._user_xlim = tuple(new_xlim)
        self._user_ylim = tuple(new_ylim)
        self._maybe_expand_grid(new_xlim, new_ylim)

    def resizeEvent(self, event):
        """窗口大小变化时重新定位3D图例标签"""
        super().resizeEvent(event)
        if hasattr(self, '_3d_legend_label') and self._3d_legend_label is not None and self._3d_legend_label.isVisible():
            gw = self._gl_widget.width() if self._gl_widget else self.width()
            lw = self._3d_legend_label.width()
            self._3d_legend_label.move(gw - lw - 8, 8)
            self._3d_legend_label.raise_()

    def compute_grid(self, pes, resolution=200):
        """计算势能面网格数据"""
        self.pes = pes
        if self._custom_bounds is not None:
            xmin, xmax, ymin, ymax = self._custom_bounds
        else:
            xmin, xmax, ymin, ymax = pes.bounds
        # 保存用户指定的原始范围（用于视图显示）
        self._original_bounds = (xmin, xmax, ymin, ymax)
        # 网格范围稍微扩大以支持缩放/平移
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        xhalf = (xmax - xmin) * 0.75  # 扩大50%
        yhalf = (ymax - ymin) * 0.75
        x = np.linspace(xcenter - xhalf, xcenter + xhalf, int(resolution * 1.5))
        y = np.linspace(ycenter - yhalf, ycenter + yhalf, int(resolution * 1.5))
        self.X, self.Y = np.meshgrid(x, y)
        try:
            self.Z = np.vectorize(pes.energy)(self.X, self.Y)
        except Exception:
            self.Z = np.zeros_like(self.X)
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        self.Z[i, j] = pes.energy(self.X[i, j], self.Y[i, j])
                    except Exception:
                        self.Z[i, j] = np.nan
        # 保存原始Z数据（无偏置）
        self._Z_original = self.Z.copy()

    def set_bias_function(self, bias_func):
        """设置偏置势函数（用于SSW/MetaDynamics等算法的可视化）

        Args:
            bias_func: 可调用函数 bias_func(x, y) -> float，或 None（清除偏置）
        """
        self._bias_function = bias_func
        if self.X is not None and self._Z_original is not None:
            if bias_func is not None:
                try:
                    bias_grid = np.vectorize(bias_func)(self.X, self.Y)
                    self.Z = self._Z_original + bias_grid
                except Exception:
                    self.Z = self._Z_original.copy()
            else:
                self.Z = self._Z_original.copy()

    def _need_layout_change(self, new_layout):
        """判断是否需要重建布局"""
        return self._current_layout != new_layout

    def _clear_contour_items(self):
        """清除2D等高线图上的所有绘图项"""
        for item in self._contour_items:
            try:
                self._contour_widget.removeItem(item)
            except Exception:
                pass
        self._contour_items = []
        self._contour_line_items = []
        if self._image_item is not None:
            try:
                self._contour_widget.removeItem(self._image_item)
            except Exception:
                pass
            self._image_item = None

    def _clear_3d_items(self):
        """清除3D视图上的所有绘图项"""
        if self._gl_widget is None:
            return
        for item in self._3d_items:
            try:
                self._gl_widget.removeItem(item)
            except Exception:
                pass
        self._3d_items = []

    def _save_camera_state(self):
        """保存当前3D相机状态"""
        if self._gl_widget is None:
            return None
        try:
            opts = self._gl_widget.opts
            return {
                'distance': opts['distance'],
                'elevation': opts['elevation'],
                'azimuth': opts['azimuth'],
                'center': opts['center'],
                'fov': opts.get('fov', 60),
            }
        except Exception:
            return None

    def _restore_camera_state(self, cam_state):
        """恢复3D相机状态"""
        if cam_state is None or self._gl_widget is None:
            # 首次显示或范围变化时自动计算合适的相机参数
            try:
                if self.X is not None and self.Z is not None:
                    x_range = self.X.max() - self.X.min()
                    y_range = self.Y.max() - self.Y.min()
                    z_scale = getattr(self, '_z_scale', 1.0)
                    z_min = getattr(self, '_z_offset', 0.0)
                    z_max_scaled = (np.nanmax(self.Z) - z_min) * z_scale
                    z_min_scaled = 0
                    z_range_scaled = z_max_scaled - z_min_scaled
                    avg_range = max(x_range, y_range, z_range_scaled)
                    # 重置相机中心到新PES的几何中心，保证旋转围绕当前图样中心
                    center_x = (self.X.min() + self.X.max()) / 2.0
                    center_y = (self.Y.min() + self.Y.max()) / 2.0
                    center_z = z_range_scaled / 2.0
                    from PyQt5.QtGui import QVector3D
                    self._gl_widget.opts['center'] = QVector3D(center_x, center_y, center_z)
                    self._gl_widget.setCameraPosition(
                        distance=avg_range * 2.0, elevation=30, azimuth=45)
            except Exception:
                pass
            return
        try:
            self._gl_widget.opts['distance'] = cam_state['distance']
            self._gl_widget.opts['elevation'] = cam_state['elevation']
            self._gl_widget.opts['azimuth'] = cam_state['azimuth']
            self._gl_widget.opts['center'] = cam_state['center']
            if 'fov' in cam_state:
                self._gl_widget.opts['fov'] = cam_state['fov']
            self._gl_widget.update()
        except Exception:
            pass

    def _add_3d_axes(self, Xs, Ys, Zs_scaled, z_min_real, z_max_real, z_scale):
        """添加三维坐标轴（粗轴线+刻度线+标签+底面网格线）

        参数:
            Xs, Ys: 2D网格坐标
            Zs_scaled: 缩放后的2D能量数据
            z_min_real, z_max_real: 真实能量范围
            z_scale: Z轴缩放因子
        """
        if not HAS_OPENGL or self._gl_widget is None:
            return

        x_min, x_max = float(Xs.min()), float(Xs.max())
        y_min, y_max = float(Ys.min()), float(Ys.max())
        z_min_s, z_max_s = float(Zs_scaled.min()), float(Zs_scaled.max())

        # 计算合适的刻度间隔
        def nice_ticks(vmin, vmax, n_ticks=5):
            """计算美观的刻度位置"""
            import math
            if vmax - vmin < 1e-10:
                return [vmin]
            raw_step = (vmax - vmin) / n_ticks
            magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
            residual = raw_step / magnitude
            if residual <= 1.5:
                nice_step = 1 * magnitude
            elif residual <= 3:
                nice_step = 2 * magnitude
            elif residual <= 7:
                nice_step = 5 * magnitude
            else:
                nice_step = 10 * magnitude
            start = math.ceil(vmin / nice_step) * nice_step
            ticks = []
            v = start
            while v <= vmax + nice_step * 0.01:
                ticks.append(round(v, 10))
                v += nice_step
            return ticks

        # ---- 粗轴线 ----
        axis_color = (0.2, 0.2, 0.2, 1.0)
        axis_width = 2.0
        # X轴
        ax = GLLinePlotItem(
            pos=np.array([[x_min, y_min, z_min_s], [x_max, y_min, z_min_s]]),
            color=axis_color, width=axis_width, antialias=True)
        self._gl_widget.addItem(ax)
        self._3d_items.append(ax)
        # Y轴
        ax = GLLinePlotItem(
            pos=np.array([[x_min, y_min, z_min_s], [x_min, y_max, z_min_s]]),
            color=axis_color, width=axis_width, antialias=True)
        self._gl_widget.addItem(ax)
        self._3d_items.append(ax)
        # Z轴
        ax = GLLinePlotItem(
            pos=np.array([[x_min, y_min, z_min_s], [x_min, y_min, z_max_s]]),
            color=axis_color, width=axis_width, antialias=True)
        self._gl_widget.addItem(ax)
        self._3d_items.append(ax)

        # ---- 底面网格线 ----
        grid_color = (0.7, 0.7, 0.7, 0.4)
        grid_width = 0.5
        x_ticks = nice_ticks(x_min, x_max, 5)
        y_ticks = nice_ticks(y_min, y_max, 5)
        # X方向网格线（平行于X轴）
        for yt in y_ticks:
            line = GLLinePlotItem(
                pos=np.array([[x_min, yt, z_min_s], [x_max, yt, z_min_s]]),
                color=grid_color, width=grid_width, antialias=True)
            self._gl_widget.addItem(line)
            self._3d_items.append(line)
        # Y方向网格线（平行于Y轴）
        for xt in x_ticks:
            line = GLLinePlotItem(
                pos=np.array([[xt, y_min, z_min_s], [xt, y_max, z_min_s]]),
                color=grid_color, width=grid_width, antialias=True)
            self._gl_widget.addItem(line)
            self._3d_items.append(line)

        # ---- 刻度线和标签 ----
        tick_color = (0.3, 0.3, 0.3, 1.0)
        label_color = (50, 50, 50, 255)

        # X轴刻度
        for xt in x_ticks:
            tick_len = (y_max - y_min) * 0.03
            line = GLLinePlotItem(
                pos=np.array([[xt, y_min, z_min_s], [xt, y_min - tick_len, z_min_s]]),
                color=tick_color, width=1.5, antialias=True)
            self._gl_widget.addItem(line)
            self._3d_items.append(line)
            text = GLTextItem(
                pos=np.array([xt, y_min - tick_len * 3, z_min_s]),
                text=f"{xt:.2g}", color=label_color)
            self._gl_widget.addItem(text)
            self._3d_items.append(text)

        # Y轴刻度
        for yt in y_ticks:
            tick_len = (x_max - x_min) * 0.03
            line = GLLinePlotItem(
                pos=np.array([[x_min, yt, z_min_s], [x_min - tick_len, yt, z_min_s]]),
                color=tick_color, width=1.5, antialias=True)
            self._gl_widget.addItem(line)
            self._3d_items.append(line)
            text = GLTextItem(
                pos=np.array([x_min - tick_len * 3, yt, z_min_s]),
                text=f"{yt:.2g}", color=label_color)
            self._gl_widget.addItem(text)
            self._3d_items.append(text)

        # Z轴刻度（显示真实能量值）
        z_ticks_real = nice_ticks(z_min_real, z_max_real, 5)
        for zt in z_ticks_real:
            zt_scaled = (zt - z_min_real) * z_scale
            if zt_scaled < z_min_s - 0.01 or zt_scaled > z_max_s + 0.01:
                continue
            tick_len = (x_max - x_min) * 0.03
            line = GLLinePlotItem(
                pos=np.array([[x_min, y_min, zt_scaled], [x_min - tick_len, y_min, zt_scaled]]),
                color=tick_color, width=1.5, antialias=True)
            self._gl_widget.addItem(line)
            self._3d_items.append(line)
            text = GLTextItem(
                pos=np.array([x_min - tick_len * 3, y_min, zt_scaled]),
                text=f"{zt:.2g}", color=label_color)
            self._gl_widget.addItem(text)
            self._3d_items.append(text)

        # ---- 轴标签 ----
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid_s = (z_min_s + z_max_s) / 2
        axis_label_color = (0, 0, 160, 255)

        x_label = GLTextItem(
            pos=np.array([x_mid, y_min - (y_max - y_min) * 0.15, z_min_s]),
            text="x", color=axis_label_color)
        self._gl_widget.addItem(x_label)
        self._3d_items.append(x_label)

        y_label = GLTextItem(
            pos=np.array([x_min - (x_max - x_min) * 0.15, y_mid, z_min_s]),
            text="y", color=axis_label_color)
        self._gl_widget.addItem(y_label)
        self._3d_items.append(y_label)

        z_label = GLTextItem(
            pos=np.array([x_min - (x_max - x_min) * 0.15, y_min, z_mid_s]),
            text="E", color=axis_label_color)
        self._gl_widget.addItem(z_label)
        self._3d_items.append(z_label)

    def _clear_energy_plot(self, widget):
        """清除能量曲线子图"""
        widget.clear()
        widget.hide()

    def _get_coolwarm_colormap(self):
        """获取coolwarm颜色映射"""
        # 使用pyqtgraph内置的colormap
        try:
            cmap = pg.colormap.get('coolwarm')
            if cmap is not None:
                return cmap
        except Exception:
            pass
        # 手动创建coolwarm
        try:
            cmap = pg.colormap.getFromMatplotlib('coolwarm')
            if cmap is not None:
                return cmap
        except Exception:
            pass
        # 回退：手动定义
        colors = [
            (59, 76, 192),   # 蓝
            (141, 176, 254),
            (221, 221, 221), # 白
            (245, 162, 127),
            (180, 4, 38),    # 红
        ]
        return pg.ColorMap(pos=np.linspace(0, 1, len(colors)), color=colors)

    def draw_contour(self, trajectory=None, start_pos=None, end_pos=None,
                     dimer_info=None, neb_images=None, energy_profile=None,
                     population_positions=None):
        """绘制等高线图"""
        self.mode = 'contour'

        if self.Z is None:
            return

        # 确定布局类型
        new_layout = 'contour+energy' if energy_profile is not None else 'contour'

        # ★ 在清除绘图项之前，无条件读取当前视图范围并更新_user_xlim/_user_ylim
        # 这确保用户缩放/平移后的视图状态不会丢失
        if not self._need_layout_change(new_layout):
            try:
                vb = self._contour_widget.plotItem.vb
                view_range = vb.viewRange()
                xlim = view_range[0]
                ylim = view_range[1]
                if (np.isfinite(xlim[0]) and np.isfinite(xlim[1]) and
                    np.isfinite(ylim[0]) and np.isfinite(ylim[1]) and
                    xlim[0] != xlim[1] and ylim[0] != ylim[1]):
                    self._user_xlim = tuple(xlim)
                    self._user_ylim = tuple(ylim)
            except Exception:
                pass

        # 清除旧绘图项
        self._clear_contour_items()

        # 切换到等高线页面
        self._stack.setCurrentIndex(0)

        # 确定视图范围
        if self._user_xlim is not None:
            plot_xlim = self._user_xlim
        elif hasattr(self, '_original_bounds') and self._original_bounds is not None:
            plot_xlim = (self._original_bounds[0], self._original_bounds[1])
        else:
            plot_xlim = None
        if self._user_ylim is not None:
            plot_ylim = self._user_ylim
        elif hasattr(self, '_original_bounds') and self._original_bounds is not None:
            plot_ylim = (self._original_bounds[2], self._original_bounds[3])
        else:
            plot_ylim = None

        # 校验Z数据
        z_min, z_max = np.nanmin(self.Z), np.nanmax(self.Z)
        if np.isnan(z_min) or np.isnan(z_max) or z_min == z_max:
            if hasattr(self, '_original_bounds'):
                xmin, xmax, ymin, ymax = self._original_bounds
            else:
                xmin, xmax, ymin, ymax = self.pes.bounds if self.pes else (-3, 3, -3, 3)
            if plot_xlim is not None:
                self._contour_widget.setXRange(plot_xlim[0], plot_xlim[1], padding=0)
            else:
                self._contour_widget.setXRange(xmin, xmax, padding=0)
            if plot_ylim is not None:
                self._contour_widget.setYRange(plot_ylim[0], plot_ylim[1], padding=0)
            else:
                self._contour_widget.setYRange(ymin, ymax, padding=0)
            self._current_layout = new_layout
            return

        # ---- 绘制PES图像 ----
        Z_plot = np.where(np.isnan(self.Z), z_min, self.Z)

        # 计算图像的变换参数
        # ImageItem使用(row, col)索引，row=y, col=x
        x_vals = self.X[0, :]
        y_vals = self.Y[:, 0]
        x_min_val, x_max_val = x_vals[0], x_vals[-1]
        y_min_val, y_max_val = y_vals[0], y_vals[-1]
        dx = (x_max_val - x_min_val) / (len(x_vals) - 1) if len(x_vals) > 1 else 1
        dy = (y_max_val - y_min_val) / (len(y_vals) - 1) if len(y_vals) > 1 else 1

        # 创建ImageItem
        self._image_item = ImageItem()
        self._image_item.setImage(Z_plot)

        # 设置变换：将像素坐标映射到数据坐标
        # pos=(x_origin, y_origin), scale=(dx_per_pixel, dy_per_pixel)
        # ImageItem的pos是左下角坐标
        self._image_item.setRect(QRectF(
            x_min_val - dx/2,
            y_min_val - dy/2,
            (x_max_val - x_min_val) + dx,
            (y_max_val - y_min_val) + dy
        ))

        # 设置颜色映射
        cmap = self._get_coolwarm_colormap()
        # 使用setColorMap方法（更可靠），同时设置LUT作为备选
        try:
            self._image_item.setColorMap(cmap)
        except Exception:
            lut = cmap.getLookupTable(nPts=256)
            self._image_item.setLookupTable(lut)
        self._image_item.setLevels([z_min, z_max])

        self._contour_widget.addItem(self._image_item)

        # 设置ColorBar（必须在addItem之前设置颜色映射，否则会被覆盖为灰度）
        try:
            # 先设置颜色映射，再关联ImageItem
            self._colorbar_widget.gradient.setColorMap(cmap)
            self._colorbar_widget.setImageItem(self._image_item)
            self._colorbar_widget.setLevels([z_min, z_max])
            # setImageItem可能会重置颜色映射，再次确保
            self._colorbar_widget.gradient.setColorMap(cmap)
        except Exception:
            pass

        # ---- 绘制等高线 ----
        if HAS_SKIMAGE:
            try:
                n_contours = 15
                levels = np.linspace(z_min, z_max, n_contours + 2)[1:-1]
                for level in levels:
                    contours = find_contours(Z_plot, level)
                    for contour in contours:
                        # 将像素坐标转为数据坐标
                        cx = x_min_val + contour[:, 1] * dx
                        cy = y_min_val + contour[:, 0] * dy
                        curve = PlotCurveItem(cx, cy, pen=pg.mkPen(color='k', width=0.5, style=Qt.DashLine))
                        self._contour_widget.addItem(curve)
                        self._contour_items.append(curve)
                        self._contour_line_items.append((curve, level))
            except Exception:
                pass

        # ---- 标注关键点 ----
        if self.pes is not None:
            cp_list = self.pes.critical_points()
            min_positions = []
            saddle_positions = []
            for cp in cp_list:
                px, py = cp['pos']
                if cp['type'] == 'minimum':
                    min_positions.append((px, py))
                elif cp['type'] == 'saddle':
                    saddle_positions.append((px, py))
            if min_positions:
                xs, ys = zip(*min_positions)
                scatter = ScatterPlotItem(x=xs, y=ys, pen=pg.mkPen('g', width=1.5),
                                         brush=pg.mkBrush('g'), size=8, name='极小值')
                self._contour_widget.addItem(scatter)
                self._contour_items.append(scatter)
            if saddle_positions:
                xs, ys = zip(*saddle_positions)
                # 用三角形表示鞍点
                scatter = ScatterPlotItem(x=xs, y=ys, pen=pg.mkPen('r', width=1.5),
                                         brush=pg.mkBrush('r'), size=10,
                                         symbol='t', name='鞍点')
                self._contour_widget.addItem(scatter)
                self._contour_items.append(scatter)

        # ---- 绘制搜索轨迹 ----
        if trajectory is not None and len(trajectory) > 0:
            if 'position' in trajectory[0]:
                path = np.array([s['position'] for s in trajectory])
            elif 'center' in trajectory[0]:
                path = np.array([s['center'] for s in trajectory])
            else:
                path = None

            if path is not None and len(path) > 0:
                # 将初始点加入路径开头，确保初始点到第一步的连线可见
                if start_pos is not None:
                    first_pt = path[0]
                    if np.linalg.norm(first_pt - start_pos) > 1e-6:
                        path = np.vstack([start_pos.reshape(1, -1), path])

                # 连线
                curve = PlotCurveItem(path[:, 0], path[:, 1],
                                      pen=pg.mkPen(color=(255, 215, 0), width=2))
                self._contour_widget.addItem(curve)
                self._contour_items.append(curve)

                # 起点：蓝色方块
                start_scatter = ScatterPlotItem(
                    x=[path[0, 0]], y=[path[0, 1]],
                    pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#2196F3'),
                    size=10, symbol='s', name='起点')
                self._contour_widget.addItem(start_scatter)
                self._contour_items.append(start_scatter)

                # 中间点
                if len(path) > 2:
                    mid_scatter = ScatterPlotItem(
                        x=path[1:-1, 0], y=path[1:-1, 1],
                        pen=pg.mkPen('w', width=0.5), brush=pg.mkBrush('#FF9800'),
                        size=5, name='中间点')
                    self._contour_widget.addItem(mid_scatter)
                    self._contour_items.append(mid_scatter)

                # 终点：红色圆圈
                end_scatter = ScatterPlotItem(
                    x=[path[-1, 0]], y=[path[-1, 1]],
                    pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#F44336'),
                    size=10, name='终点')
                self._contour_widget.addItem(end_scatter)
                self._contour_items.append(end_scatter)

        # ---- Dimer双子方向 ----
        if dimer_info is not None:
            center = dimer_info.get('center')
            direction = dimer_info.get('direction')
            delta_r = dimer_info.get('delta_r', 0.01)
            if center is not None and direction is not None:
                scale = delta_r * 5
                dx_d, dy_d = direction[0] * scale, direction[1] * scale
                # 绘制双向箭头线
                x1, y1 = center[0] - dx_d, center[1] - dy_d
                x2, y2 = center[0] + dx_d, center[1] + dy_d
                arrow = PlotCurveItem([x1, x2], [y1, y2],
                                      pen=pg.mkPen(color='m', width=2))
                self._contour_widget.addItem(arrow)
                self._contour_items.append(arrow)
                # 中心点
                center_scatter = ScatterPlotItem(
                    x=[center[0]], y=[center[1]],
                    pen=pg.mkPen('m', width=1), brush=pg.mkBrush('m'),
                    size=10, name='Dimer中心')
                self._contour_widget.addItem(center_scatter)
                self._contour_items.append(center_scatter)

        # ---- NEB弹性带 ----
        if neb_images is not None and len(neb_images) > 0:
            images = np.array(neb_images)
            neb_curve = PlotCurveItem(images[:, 0], images[:, 1],
                                      pen=pg.mkPen(color='c', width=2),
                                      connect='finite')
            self._contour_widget.addItem(neb_curve)
            self._contour_items.append(neb_curve)
            neb_scatter = ScatterPlotItem(
                x=images[:, 0], y=images[:, 1],
                pen=pg.mkPen('c', width=1), brush=pg.mkBrush('c'),
                size=7, name='弹性带')
            self._contour_widget.addItem(neb_scatter)
            self._contour_items.append(neb_scatter)

            # 标记CI点
            ci_idx = None
            if energy_profile is not None and self.pes is not None:
                ci_idx = int(np.argmax([self.pes.energy(img[0], img[1]) for img in neb_images]))
            if ci_idx is not None and 0 < ci_idx < len(neb_images) - 1:
                ci_scatter = ScatterPlotItem(
                    x=[images[ci_idx, 0]], y=[images[ci_idx, 1]],
                    pen=pg.mkPen('r', width=1.5), brush=pg.mkBrush('r'),
                    size=14, symbol='t', name='CI点')
                self._contour_widget.addItem(ci_scatter)
                self._contour_items.append(ci_scatter)

        # ---- 群体算法粒子/种群显示 ----
        if population_positions is not None and len(population_positions) > 0:
            pop = np.array(population_positions)
            pop_scatter = ScatterPlotItem(
                x=pop[:, 0], y=pop[:, 1],
                pen=pg.mkPen('w', width=0.5), brush=pg.mkBrush('#9C27B0'),
                size=6, name='种群')
            self._contour_widget.addItem(pop_scatter)
            self._contour_items.append(pop_scatter)

        # ---- 当没有轨迹时，显示初始点标记 ----
        if (trajectory is None or len(trajectory) == 0):
            if start_pos is not None:
                s = ScatterPlotItem(
                    x=[start_pos[0]], y=[start_pos[1]],
                    pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#2196F3'),
                    size=10, symbol='s', name='初始点')
                self._contour_widget.addItem(s)
                self._contour_items.append(s)
            if end_pos is not None:
                s = ScatterPlotItem(
                    x=[end_pos[0]], y=[end_pos[1]],
                    pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#F44336'),
                    size=10, symbol='t', name='终态')
                self._contour_widget.addItem(s)
                self._contour_items.append(s)

        # ---- 设置视图范围 ----
        if plot_xlim is not None:
            self._contour_widget.setXRange(plot_xlim[0], plot_xlim[1], padding=0)
        elif hasattr(self, '_original_bounds') and self._original_bounds is not None:
            self._contour_widget.setXRange(self._original_bounds[0], self._original_bounds[1], padding=0)
        if plot_ylim is not None:
            self._contour_widget.setYRange(plot_ylim[0], plot_ylim[1], padding=0)
        elif hasattr(self, '_original_bounds') and self._original_bounds is not None:
            self._contour_widget.setYRange(self._original_bounds[2], self._original_bounds[3], padding=0)

        # ---- 添加半透明图例（右上角）----
        self._add_contour_legend(trajectory is not None and len(trajectory) > 0,
                                  neb_images is not None and len(neb_images) > 0,
                                  population_positions is not None and len(population_positions) > 0)

        # ---- 能量曲线子图 ----
        if energy_profile is not None:
            distances, energies = energy_profile
            self._energy_contour_widget.show()
            self._energy_contour_widget.clear()
            self._energy_contour_widget.plot(distances, energies, pen=pg.mkPen('b', width=2),
                                              symbol='o', symbolSize=5, symbolBrush='b')
            # 标注最高能量点
            max_idx = np.argmax(energies)
            self._energy_contour_widget.plot([distances[max_idx]], [energies[max_idx]],
                                              pen=None, symbol='t', symbolSize=12,
                                              symbolBrush='r', name=f'最高能量 E={energies[max_idx]:.4f}')
            # 标注能垒水平线
            e_start = energies[0]
            barrier_line = PlotCurveItem(
                [distances[0], distances[-1]], [e_start, e_start],
                pen=pg.mkPen('g', width=1, style=Qt.DashLine))
            self._energy_contour_widget.addItem(barrier_line)
        else:
            self._clear_energy_plot(self._energy_contour_widget)

        self._current_layout = new_layout

    def _add_contour_legend(self, has_traj, has_neb, has_population):
        """在等高线图右上角添加半透明图例框，标注各类点的类型"""
        # 清除旧图例（使用更彻底的方式删除）
        if hasattr(self, '_legend_item') and self._legend_item is not None:
            try:
                # 先关闭再移除，避免残留
                self._legend_item.setVisible(False)
                # 移除所有子项
                for item in list(self._legend_item.items):
                    try:
                        self._legend_item.removeItem(item[0])
                    except Exception:
                        pass
                self._legend_item.setParentItem(None)
                if self._legend_item.scene() is not None:
                    self._legend_item.scene().removeItem(self._legend_item)
            except Exception:
                pass
            self._legend_item = None

        from pyqtgraph.graphicsItems.LegendItem import LegendItem
        legend = LegendItem(offset=(0, 0))
        legend.setParentItem(self._contour_widget.plotItem)
        # 设置半透明背景
        legend.setBrush(pg.mkBrush(255, 255, 255, 180))
        legend.setPen(pg.mkPen(100, 100, 100, 200, width=1))

        # 极小值（绿色圆点）
        if self.pes is not None and any(cp['type'] == 'minimum' for cp in self.pes.critical_points()):
            min_item = ScatterPlotItem(pen=pg.mkPen('g', width=1.5), brush=pg.mkBrush('g'),
                                        size=8, symbol='o')
            legend.addItem(min_item, '极小值')
        # 鞍点（红色三角）
        if self.pes is not None and any(cp['type'] == 'saddle' for cp in self.pes.critical_points()):
            sad_item = ScatterPlotItem(pen=pg.mkPen('r', width=1.5), brush=pg.mkBrush('r'),
                                        size=10, symbol='t')
            legend.addItem(sad_item, '鞍点')
        # 起点（蓝色方块）
        if has_traj:
            start_item = ScatterPlotItem(pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#2196F3'),
                                          size=10, symbol='s')
            legend.addItem(start_item, '起点')
            # 中间点（橙色小点）
            mid_item = ScatterPlotItem(pen=pg.mkPen('w', width=0.5), brush=pg.mkBrush('#FF9800'),
                                        size=5, symbol='o')
            legend.addItem(mid_item, '轨迹点')
            # 终点（红色圆）
            end_item = ScatterPlotItem(pen=pg.mkPen('w', width=1), brush=pg.mkBrush('#F44336'),
                                        size=10, symbol='o')
            legend.addItem(end_item, '终点')
        # NEB弹性带（青色）
        if has_neb:
            neb_item = ScatterPlotItem(pen=pg.mkPen('c', width=1), brush=pg.mkBrush('c'),
                                        size=7, symbol='o')
            legend.addItem(neb_item, '弹性带')
        # 种群（紫色）
        if has_population:
            pop_item = ScatterPlotItem(pen=pg.mkPen('w', width=0.5), brush=pg.mkBrush('#9C27B0'),
                                        size=6, symbol='o')
            legend.addItem(pop_item, '种群')

        # 定位到右上角
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0))
        # 禁用拖拽，固定在右上角
        legend.setFlag(legend.ItemIsMovable, False)
        self._legend_item = legend

    def _update_3d_legend(self, has_traj, has_neb, has_population):
        """在3D视图右上角更新半透明图例标签"""
        if not hasattr(self, '_3d_legend_label') or self._3d_legend_label is None:
            return
        lines = []
        # 极小值（绿色圆点）
        if self.pes is not None and any(cp['type'] == 'minimum' for cp in self.pes.critical_points()):
            lines.append('<span style="color:#2e7d32;">●</span> 极小值')
        # 鞍点（红色三角）
        if self.pes is not None and any(cp['type'] == 'saddle' for cp in self.pes.critical_points()):
            lines.append('<span style="color:#c62828;">▲</span> 鞍点')
        # 起点（蓝色方块）
        if has_traj:
            lines.append('<span style="color:#2196F3;">■</span> 起点')
            # 中间点（橙色小点）
            lines.append('<span style="color:#FF9800;">●</span> 轨迹点')
            # 终点（红色圆）
            lines.append('<span style="color:#F44336;">●</span> 终点')
        # NEB弹性带（青色）
        if has_neb:
            lines.append('<span style="color:#00BCD4;">●</span> 弹性带')
        # 种群（紫色）
        if has_population:
            lines.append('<span style="color:#9C27B0;">●</span> 种群')

        if lines:
            self._3d_legend_label.setText("<br>".join(lines))
            self._3d_legend_label.adjustSize()
            self._3d_legend_label.setVisible(True)
            # 定位到右上角
            gw = self._gl_widget.width()
            lw = self._3d_legend_label.width()
            self._3d_legend_label.move(gw - lw - 8, 8)
            self._3d_legend_label.raise_()
        else:
            self._3d_legend_label.setVisible(False)

    def draw_3d(self, trajectory=None, start_pos=None, end_pos=None,
                neb_images=None, energy_profile=None, population_positions=None):
        """绘制3D曲面图"""
        self.mode = '3d'

        if self.Z is None:
            return

        if not HAS_OPENGL or self._gl_widget is None:
            return

        # 确定布局类型
        new_layout = '3d+energy' if energy_profile is not None else '3d'

        # ★ 保存相机状态（在清除绘图项之前）
        # 如果范围变化需要重置相机中心，则不保存旧中心
        if getattr(self, '_reset_camera_center', False):
            cam_state = None
            self._reset_camera_center = False
        else:
            cam_state = self._save_camera_state()

        # 清除旧绘图项
        self._clear_3d_items()

        # 切换到3D页面
        self._stack.setCurrentIndex(1)

        # 校验Z数据
        z_min, z_max = np.nanmin(self.Z), np.nanmax(self.Z)
        if np.isnan(z_min) or np.isnan(z_max) or z_min == z_max:
            self._current_layout = new_layout
            return

        # ---- 创建3D网格 ----
        Z_plot = np.where(np.isnan(self.Z), z_min, self.Z)

        # 降采样以提升性能（使用更细的步长获得更光滑的曲面）
        stride = max(1, self.X.shape[0] // 120)
        Xs = self.X[::stride, ::stride]
        Ys = self.Y[::stride, ::stride]
        Zs = Z_plot[::stride, ::stride]

        # ★ 缩放Z轴数据，使曲面起伏在视觉上与XY轴匹配
        x_range = Xs.max() - Xs.min()
        y_range = Ys.max() - Ys.min()
        z_range = z_max - z_min
        xy_avg = (x_range + y_range) / 2.0
        if z_range > 0 and xy_avg > 0:
            z_scale = xy_avg / z_range
        else:
            z_scale = 1.0
        Zs_scaled = (Zs - z_min) * z_scale
        self._z_scale = z_scale
        self._z_offset = z_min

        # ---- 使用GLSurfacePlotItem绘制光滑曲面 ----
        # GLSurfacePlotItem期望z[i,j]对应(x[i], y[j])，
        # 但我们的Zs[i,j]对应(x_1d[j], y_1d[i])，需要转置
        cmap = self._get_coolwarm_colormap()
        # 为每个顶点计算颜色（使用转置前的Zs_scaled）
        z_norm = (Zs_scaled.ravel() - Zs_scaled.min()) / (Zs_scaled.max() - Zs_scaled.min() + 1e-10)
        colors = cmap.map(z_norm)  # uint8 RGBA (N, 4)
        colors_float = colors.astype(float) / 255.0
        colors_float[:, 3] = 0.85  # alpha
        # reshape为 (ny, nx, 4) 然后转置为 (nx, ny, 4) 以匹配GLSurfacePlotItem
        ny, nx = Xs.shape
        colors_2d = colors_float.reshape(ny, nx, 4).transpose(1, 0, 2)

        # GLSurfacePlotItem需要x和y为1D数组，z为2D数组
        x_1d = Xs[0, :]  # 第一行的x坐标
        y_1d = Ys[:, 0]  # 第一列的y坐标
        # 转置Zs_scaled：从Zs[y_idx, x_idx]变为Zs_t[x_idx, y_idx]
        Zs_scaled_t = Zs_scaled.T

        surface = GLSurfacePlotItem(
            x=x_1d, y=y_1d, z=Zs_scaled_t,
            colors=colors_2d,
            shader=None,
            smooth=True,
            glOptions='opaque',
        )
        self._gl_widget.addItem(surface)
        self._3d_items.append(surface)

        # ---- 添加三维坐标轴 ----
        self._add_3d_axes(Xs, Ys, Zs_scaled, z_min, z_max, z_scale)

        # ---- 标注关键点 ----
        if self.pes is not None:
            for cp in self.pes.critical_points():
                px, py = cp['pos']
                pz = cp['energy']
                pz_scaled = (pz - self._z_offset) * self._z_scale
                if cp['type'] == 'minimum':
                    color = (0, 1, 0, 1)
                    size = 8
                elif cp['type'] == 'saddle':
                    color = (1, 0, 0, 1)
                    size = 10
                else:
                    continue
                scatter = GLScatterPlotItem(
                    pos=np.array([[px, py, pz_scaled]]),
                    color=np.array([color]),
                    size=size)
                self._gl_widget.addItem(scatter)
                self._3d_items.append(scatter)

        # ---- 绘制轨迹 ----
        if trajectory is not None and len(trajectory) > 0:
            if 'position' in trajectory[0]:
                path = np.array([s['position'] for s in trajectory])
                energies = np.array([s['energy'] for s in trajectory])
            elif 'center' in trajectory[0]:
                path = np.array([s['center'] for s in trajectory])
                energies = np.array([s['energy'] for s in trajectory])
            else:
                path = None
                energies = None

            if path is not None and len(path) > 0:
                # 将初始点加入路径开头，确保初始点到第一步的连线可见
                if start_pos is not None:
                    first_pt = path[0]
                    if np.linalg.norm(first_pt - start_pos) > 1e-6:
                        start_e = self.pes.energy(start_pos[0], start_pos[1]) if self.pes else 0
                        path = np.vstack([start_pos.reshape(1, -1), path])
                        energies = np.concatenate([[start_e], energies])

                energies_scaled = (energies - self._z_offset) * self._z_scale
                pts = np.column_stack([path, energies_scaled])
                line = GLLinePlotItem(pos=pts, color=(1, 0.84, 0, 1), width=2.5, antialias=True)
                self._gl_widget.addItem(line)
                self._3d_items.append(line)

                start_s = GLScatterPlotItem(
                    pos=np.array([[path[0, 0], path[0, 1], energies_scaled[0]]]),
                    color=np.array([(0.13, 0.59, 0.95, 1)]),
                    size=12)
                self._gl_widget.addItem(start_s)
                self._3d_items.append(start_s)

                # 中间点：橙色小点
                if len(path) > 2:
                    mid_pts = np.column_stack([path[1:-1], energies_scaled[1:-1]])
                    mid_s = GLScatterPlotItem(
                        pos=mid_pts,
                        color=np.array([(1.0, 0.6, 0.0, 1)] * len(mid_pts)),
                        size=6)
                    self._gl_widget.addItem(mid_s)
                    self._3d_items.append(mid_s)

                end_s = GLScatterPlotItem(
                    pos=np.array([[path[-1, 0], path[-1, 1], energies_scaled[-1]]]),
                    color=np.array([(0.96, 0.26, 0.21, 1)]),
                    size=12)
                self._gl_widget.addItem(end_s)
                self._3d_items.append(end_s)

        # ---- NEB弹性带 ----
        if neb_images is not None and len(neb_images) > 0:
            images = np.array(neb_images)
            neb_energies = np.array([self.pes.energy(img[0], img[1]) for img in neb_images])
            neb_energies_scaled = (neb_energies - self._z_offset) * self._z_scale
            pts = np.column_stack([images, neb_energies_scaled])
            neb_line = GLLinePlotItem(pos=pts, color=(0, 1, 1, 1), width=2, antialias=True)
            self._gl_widget.addItem(neb_line)
            self._3d_items.append(neb_line)
            neb_scatter = GLScatterPlotItem(pos=pts, color=(0, 1, 1, 1), size=6)
            self._gl_widget.addItem(neb_scatter)
            self._3d_items.append(neb_scatter)

        # ---- 群体算法粒子/种群显示 ----
        if population_positions is not None and len(population_positions) > 0:
            pop = np.array(population_positions)
            pop_energies = np.array([self.pes.energy(p[0], p[1]) for p in pop])
            pop_energies_scaled = (pop_energies - self._z_offset) * self._z_scale
            pop_pts = np.column_stack([pop, pop_energies_scaled])
            pop_scatter = GLScatterPlotItem(
                pos=pop_pts,
                color=np.array([(0.61, 0.15, 0.69, 1)] * len(pop)),  # 紫色
                size=6)
            self._gl_widget.addItem(pop_scatter)
            self._3d_items.append(pop_scatter)

        # ---- 当没有轨迹时，显示初始点标记 ----
        if (trajectory is None or len(trajectory) == 0):
            if start_pos is not None:
                e = self.pes.energy(start_pos[0], start_pos[1]) if self.pes else 0
                e_scaled = (e - self._z_offset) * self._z_scale if self.pes else 0
                s = GLScatterPlotItem(
                    pos=np.array([[start_pos[0], start_pos[1], e_scaled]]),
                    color=np.array([(0.13, 0.59, 0.95, 1)]),
                    size=12)
                self._gl_widget.addItem(s)
                self._3d_items.append(s)
            if end_pos is not None:
                e = self.pes.energy(end_pos[0], end_pos[1]) if self.pes else 0
                e_scaled = (e - self._z_offset) * self._z_scale if self.pes else 0
                s = GLScatterPlotItem(
                    pos=np.array([[end_pos[0], end_pos[1], e_scaled]]),
                    color=np.array([(0.96, 0.26, 0.21, 1)]),
                    size=12)
                self._gl_widget.addItem(s)
                self._3d_items.append(s)

        # ---- 恢复相机状态 ----
        self._restore_camera_state(cam_state)

        # ---- 更新3D图例 ----
        has_traj = trajectory is not None and len(trajectory) > 0
        has_neb = neb_images is not None and len(neb_images) > 0
        has_pop = population_positions is not None and len(population_positions) > 0
        self._update_3d_legend(has_traj, has_neb, has_pop)

        # ---- 能量曲线子图 ----
        if energy_profile is not None:
            distances, energies = energy_profile
            self._energy_3d_widget.show()
            self._energy_3d_widget.clear()
            self._energy_3d_widget.plot(distances, energies, pen=pg.mkPen('b', width=2),
                                         symbol='o', symbolSize=5, symbolBrush='b')
            max_idx = np.argmax(energies)
            self._energy_3d_widget.plot([distances[max_idx]], [energies[max_idx]],
                                         pen=None, symbol='t', symbolSize=12,
                                         symbolBrush='r', name=f'最高能量 E={energies[max_idx]:.4f}')
            e_start = energies[0]
            barrier_line = PlotCurveItem(
                [distances[0], distances[-1]], [e_start, e_start],
                pen=pg.mkPen('g', width=1, style=Qt.DashLine))
            self._energy_3d_widget.addItem(barrier_line)
        else:
            self._clear_energy_plot(self._energy_3d_widget)

        self._current_layout = new_layout

    def set_view_range(self, xlim, ylim):
        """设置视图范围（用于GIF导出兼容）"""
        if self.mode == 'contour':
            self._contour_widget.setXRange(xlim[0], xlim[1], padding=0)
            self._contour_widget.setYRange(ylim[0], ylim[1], padding=0)
        # 3D模式不支持固定2D范围

    def refresh(self):
        """刷新显示（兼容方法）"""
        self.update()

    def save_png(self, filepath):
        """保存当前图形为PNG"""
        pixmap = self.grab()
        pixmap.save(filepath)

    def get_frame_image(self):
        """获取当前帧的图像数据（用于GIF导出）"""
        pixmap = self.grab()
        image = pixmap.toImage()
        # QImage.save()不支持BytesIO，需要使用QBuffer
        from PyQt5.QtCore import QBuffer
        buf = QBuffer()
        buf.open(QBuffer.ReadWrite)
        image.save(buf, 'PNG')
        buf.seek(0)
        data = buf.data()
        buf.close()
        import io
        return io.BytesIO(data.data())


class _ViewRangeProxy:
    """兼容性代理类，提供 set_xlim/set_ylim 方法供GIF导出使用"""

    def __init__(self, canvas):
        self._canvas = canvas

    def set_xlim(self, xlim):
        """设置X轴范围"""
        if self._canvas.mode == 'contour':
            self._canvas._contour_widget.setXRange(xlim[0], xlim[1], padding=0)
            self._canvas._user_xlim = tuple(xlim)

    def set_ylim(self, ylim):
        """设置Y轴范围"""
        if self._canvas.mode == 'contour':
            self._canvas._contour_widget.setYRange(ylim[0], ylim[1], padding=0)
            self._canvas._user_ylim = tuple(ylim)


# ======================== 左侧控制面板 ========================

class ControlPanel(QWidget):
    """左侧控制面板：算法选择、参数调节、播放控制"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        # ---- 势能面选择区 ----
        pes_group = QGroupBox("势能面选择")
        pes_group.setProperty("flat", True)
        pes_layout = QVBoxLayout()
        pes_layout.setSpacing(2)
        pes_layout.setContentsMargins(4, 4, 4, 4)

        pes_type_layout = QHBoxLayout()
        pes_type_layout.addWidget(QLabel("类型:"))
        self.pes_combo = QComboBox()
        self.pes_combo.addItems(["Müller-Brown", "双阱", "三极小值", "Rosenbrock", "Himmelblau", "Ackley", "Rastrigin", "自定义"])
        pes_type_layout.addWidget(self.pes_combo)
        pes_layout.addLayout(pes_type_layout)

        self.custom_expr_label = QLabel("表达式(变量x,y; 幂用^或**; 支持2x,2sin等简写):")
        self.custom_expr_label.setVisible(False)
        self.custom_expr = QLineEdit("(x**2-1)**2+y**2")
        self.custom_expr.setVisible(False)
        self.custom_expr_confirm_btn = QPushButton("确认")
        self.custom_expr_confirm_btn.setObjectName("customConfirmBtn")
        self.custom_expr_confirm_btn.setVisible(False)
        self.custom_expr_confirm_btn.setFixedHeight(24)
        custom_expr_layout = QHBoxLayout()
        custom_expr_layout.addWidget(self.custom_expr, 1)
        custom_expr_layout.addWidget(self.custom_expr_confirm_btn, 0)
        pes_layout.addWidget(self.custom_expr_label)
        pes_layout.addLayout(custom_expr_layout)

        mode_layout = QHBoxLayout()
        self.contour_check = QCheckBox("等高线")
        self.contour_check.setChecked(True)
        self.surface3d_check = QCheckBox("3D曲面")
        mode_layout.addWidget(self.contour_check)
        mode_layout.addWidget(self.surface3d_check)
        mode_layout.addStretch()
        pes_layout.addLayout(mode_layout)

        # PES范围选择
        range_label = QLabel("显示范围:")
        pes_layout.addWidget(range_label)
        range_x_layout = QHBoxLayout()
        range_x_layout.addWidget(QLabel("x:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setDecimals(2)
        self.x_min_spin.setRange(-100, 100)
        self.x_min_spin.setValue(-1.5)
        self.x_min_spin.setSingleStep(0.1)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setDecimals(2)
        self.x_max_spin.setRange(-100, 100)
        self.x_max_spin.setValue(1.5)
        self.x_max_spin.setSingleStep(0.1)
        range_x_layout.addWidget(self.x_min_spin)
        range_x_layout.addWidget(QLabel("~"))
        range_x_layout.addWidget(self.x_max_spin)
        pes_layout.addLayout(range_x_layout)

        # x范围滑块
        self.x_min_slider = QSlider(Qt.Horizontal)
        self.x_min_slider.setRange(-10000, 10000)
        self.x_min_slider.setValue(int(-1.5 * 100))
        self.x_max_slider = QSlider(Qt.Horizontal)
        self.x_max_slider.setRange(-10000, 10000)
        self.x_max_slider.setValue(int(1.5 * 100))
        x_slider_layout = QHBoxLayout()
        x_slider_layout.addWidget(self.x_min_slider)
        x_slider_layout.addWidget(self.x_max_slider)
        pes_layout.addLayout(x_slider_layout)

        range_y_layout = QHBoxLayout()
        range_y_layout.addWidget(QLabel("y:"))
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setDecimals(2)
        self.y_min_spin.setRange(-100, 100)
        self.y_min_spin.setValue(-0.5)
        self.y_min_spin.setSingleStep(0.1)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setDecimals(2)
        self.y_max_spin.setRange(-100, 100)
        self.y_max_spin.setValue(2.5)
        self.y_max_spin.setSingleStep(0.1)
        range_y_layout.addWidget(self.y_min_spin)
        range_y_layout.addWidget(QLabel("~"))
        range_y_layout.addWidget(self.y_max_spin)
        pes_layout.addLayout(range_y_layout)

        # y范围滑块
        self.y_min_slider = QSlider(Qt.Horizontal)
        self.y_min_slider.setRange(-10000, 10000)
        self.y_min_slider.setValue(int(-0.5 * 100))
        self.y_max_slider = QSlider(Qt.Horizontal)
        self.y_max_slider.setRange(-10000, 10000)
        self.y_max_slider.setValue(int(2.5 * 100))
        y_slider_layout = QHBoxLayout()
        y_slider_layout.addWidget(self.y_min_slider)
        y_slider_layout.addWidget(self.y_max_slider)
        pes_layout.addLayout(y_slider_layout)

        # 滑块与spinbox双向同步
        self.x_min_slider.valueChanged.connect(lambda v: self.x_min_spin.setValue(v / 100.0))
        self.x_max_slider.valueChanged.connect(lambda v: self.x_max_spin.setValue(v / 100.0))
        self.y_min_slider.valueChanged.connect(lambda v: self.y_min_spin.setValue(v / 100.0))
        self.y_max_slider.valueChanged.connect(lambda v: self.y_max_spin.setValue(v / 100.0))
        self.x_min_spin.valueChanged.connect(lambda v: self.x_min_slider.setValue(int(v * 100)))
        self.x_max_spin.valueChanged.connect(lambda v: self.x_max_slider.setValue(int(v * 100)))
        self.y_min_spin.valueChanged.connect(lambda v: self.y_min_slider.setValue(int(v * 100)))
        self.y_max_spin.valueChanged.connect(lambda v: self.y_max_slider.setValue(int(v * 100)))

        self.apply_range_btn = QPushButton("应用范围")
        self.apply_range_btn.setObjectName("applyRangeBtn")
        self.apply_range_btn.setFixedHeight(24)
        pes_layout.addWidget(self.apply_range_btn)

        pes_group.setLayout(pes_layout)
        layout.addWidget(pes_group)

        # ---- 算法选择区 ----
        algo_group = QGroupBox("算法选择")
        algo_group.setProperty("flat", True)
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(2)
        algo_layout.setContentsMargins(4, 4, 4, 4)

        algo_sel_layout = QHBoxLayout()
        algo_sel_layout.addWidget(QLabel("算法:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "牛顿法(极小值)", "牛顿法(鞍点)", "Dimer方法",
            "NEB方法", "CI-NEB方法", "最速下降法", "盆地跳跃法", "Metropolis MC",
            "SSW方法", "元动力学", "极小点跳跃", "遗传算法", "粒子群优化",
            "人工蜂群", "伞形采样", "自适应偏置力", "CBD方法", "DESW方法"
        ])
        algo_sel_layout.addWidget(self.algo_combo)
        algo_layout.addLayout(algo_sel_layout)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # ---- 牛顿法参数区 ----
        self.newton_group = QGroupBox("牛顿法参数")
        self.newton_group.setProperty("flat", True)
        newton_layout = QVBoxLayout()
        newton_layout.setSpacing(2)
        newton_layout.setContentsMargins(4, 4, 4, 4)

        thr_layout = QHBoxLayout()
        thr_layout.addWidget(QLabel("收敛阈值:"))
        self.newton_threshold = QDoubleSpinBox()
        self.newton_threshold.setRange(1e-6, 1e-1)
        self.newton_threshold.setValue(1e-4)
        self.newton_threshold.setDecimals(6)
        self.newton_threshold.setSingleStep(1e-5)
        thr_layout.addWidget(self.newton_threshold)
        newton_layout.addLayout(thr_layout)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("最大迭代:"))
        self.newton_max_iter = QSpinBox()
        self.newton_max_iter.setRange(10, 500)
        self.newton_max_iter.setValue(100)
        iter_layout.addWidget(self.newton_max_iter)
        newton_layout.addLayout(iter_layout)

        self.newton_confirm_btn = QPushButton("确认参数")
        self.newton_confirm_btn.setObjectName("applyRangeBtn")
        self.newton_confirm_btn.setFixedHeight(22)
        newton_layout.addWidget(self.newton_confirm_btn)

        self.newton_group.setLayout(newton_layout)
        layout.addWidget(self.newton_group)

        # ---- Dimer方法参数区 ----
        self.dimer_group = QGroupBox("Dimer方法参数")
        self.dimer_group.setProperty("flat", True)
        dimer_layout = QVBoxLayout()
        dimer_layout.setSpacing(2)
        dimer_layout.setContentsMargins(4, 4, 4, 4)

        dr_layout = QHBoxLayout()
        dr_layout.addWidget(QLabel("双子间距 ΔR:"))
        self.dimer_delta_r = QDoubleSpinBox()
        self.dimer_delta_r.setRange(0.001, 0.1)
        self.dimer_delta_r.setValue(0.01)
        self.dimer_delta_r.setDecimals(3)
        self.dimer_delta_r.setSingleStep(0.001)
        dr_layout.addWidget(self.dimer_delta_r)
        dimer_layout.addLayout(dr_layout)

        dthr_layout = QHBoxLayout()
        dthr_layout.addWidget(QLabel("收敛阈值:"))
        self.dimer_threshold = QDoubleSpinBox()
        self.dimer_threshold.setRange(1e-4, 1e-1)
        self.dimer_threshold.setValue(1e-3)
        self.dimer_threshold.setDecimals(4)
        self.dimer_threshold.setSingleStep(1e-4)
        dthr_layout.addWidget(self.dimer_threshold)
        dimer_layout.addLayout(dthr_layout)

        diter_layout = QHBoxLayout()
        diter_layout.addWidget(QLabel("最大迭代:"))
        self.dimer_max_iter = QSpinBox()
        self.dimer_max_iter.setRange(50, 1000)
        self.dimer_max_iter.setValue(200)
        diter_layout.addWidget(self.dimer_max_iter)
        dimer_layout.addLayout(diter_layout)

        dstep_layout = QHBoxLayout()
        dstep_layout.addWidget(QLabel("平动步长:"))
        self.dimer_trans_step = QDoubleSpinBox()
        self.dimer_trans_step.setRange(0.01, 1.0)
        self.dimer_trans_step.setValue(0.05)
        self.dimer_trans_step.setDecimals(2)
        self.dimer_trans_step.setSingleStep(0.01)
        dstep_layout.addWidget(self.dimer_trans_step)
        dimer_layout.addLayout(dstep_layout)

        self.dimer_confirm_btn = QPushButton("确认参数")
        self.dimer_confirm_btn.setObjectName("applyRangeBtn")
        self.dimer_confirm_btn.setFixedHeight(22)
        dimer_layout.addWidget(self.dimer_confirm_btn)

        self.dimer_group.setLayout(dimer_layout)
        layout.addWidget(self.dimer_group)

        # ---- NEB/CI-NEB参数区 ----
        self.neb_group = QGroupBox("NEB/CI-NEB参数")
        self.neb_group.setProperty("flat", True)
        neb_layout = QVBoxLayout()
        neb_layout.setSpacing(2)
        neb_layout.setContentsMargins(4, 4, 4, 4)

        nimg_layout = QHBoxLayout()
        nimg_layout.addWidget(QLabel("镜像点数:"))
        self.neb_n_images = QSpinBox()
        self.neb_n_images.setRange(3, 30)
        self.neb_n_images.setValue(10)
        nimg_layout.addWidget(self.neb_n_images)
        neb_layout.addLayout(nimg_layout)

        sk_layout = QHBoxLayout()
        sk_layout.addWidget(QLabel("弹簧力常数:"))
        self.neb_spring_k = QDoubleSpinBox()
        self.neb_spring_k.setRange(0.01, 10.0)
        self.neb_spring_k.setValue(1.0)
        self.neb_spring_k.setDecimals(2)
        self.neb_spring_k.setSingleStep(0.1)
        sk_layout.addWidget(self.neb_spring_k)
        neb_layout.addLayout(sk_layout)

        nthr_layout = QHBoxLayout()
        nthr_layout.addWidget(QLabel("收敛阈值:"))
        self.neb_threshold = QDoubleSpinBox()
        self.neb_threshold.setRange(1e-4, 1e-1)
        self.neb_threshold.setValue(1e-3)
        self.neb_threshold.setDecimals(4)
        self.neb_threshold.setSingleStep(1e-4)
        nthr_layout.addWidget(self.neb_threshold)
        neb_layout.addLayout(nthr_layout)

        niter_layout = QHBoxLayout()
        niter_layout.addWidget(QLabel("最大优化步:"))
        self.neb_max_iter = QSpinBox()
        self.neb_max_iter.setRange(50, 1000)
        self.neb_max_iter.setValue(200)
        niter_layout.addWidget(self.neb_max_iter)
        neb_layout.addLayout(niter_layout)

        self.neb_confirm_btn = QPushButton("确认参数")
        self.neb_confirm_btn.setObjectName("applyRangeBtn")
        self.neb_confirm_btn.setFixedHeight(22)
        neb_layout.addWidget(self.neb_confirm_btn)

        self.neb_group.setLayout(neb_layout)
        layout.addWidget(self.neb_group)

        # 最速下降法参数
        self.sd_group = QGroupBox("最速下降法参数")
        self.sd_group.setProperty("flat", True)
        sd_layout = QVBoxLayout()
        sd_layout.setSpacing(2)
        sd_layout.setContentsMargins(4, 4, 4, 4)

        sd_step_layout = QHBoxLayout()
        sd_step_layout.addWidget(QLabel("步长:"))
        self.sd_step_size = QDoubleSpinBox()
        self.sd_step_size.setRange(0.001, 1.0)
        self.sd_step_size.setValue(0.05)
        self.sd_step_size.setDecimals(3)
        self.sd_step_size.setSingleStep(0.01)
        sd_step_layout.addWidget(self.sd_step_size)
        sd_layout.addLayout(sd_step_layout)

        sd_thr_layout = QHBoxLayout()
        sd_thr_layout.addWidget(QLabel("收敛阈值:"))
        self.sd_threshold = QDoubleSpinBox()
        self.sd_threshold.setRange(1e-6, 1e-1)
        self.sd_threshold.setValue(1e-4)
        self.sd_threshold.setDecimals(6)
        self.sd_threshold.setSingleStep(1e-5)
        sd_thr_layout.addWidget(self.sd_threshold)
        sd_layout.addLayout(sd_thr_layout)

        sd_iter_layout = QHBoxLayout()
        sd_iter_layout.addWidget(QLabel("最大迭代:"))
        self.sd_max_iter = QSpinBox()
        self.sd_max_iter.setRange(10, 1000)
        self.sd_max_iter.setValue(200)
        sd_iter_layout.addWidget(self.sd_max_iter)
        sd_layout.addLayout(sd_iter_layout)

        self.sd_confirm_btn = QPushButton("确认参数")
        self.sd_confirm_btn.setObjectName("applyRangeBtn")
        self.sd_confirm_btn.setFixedHeight(22)
        sd_layout.addWidget(self.sd_confirm_btn)

        self.sd_group.setLayout(sd_layout)
        layout.addWidget(self.sd_group)

        # 盆地跳跃法参数
        self.bh_group = QGroupBox("盆地跳跃法参数")
        self.bh_group.setProperty("flat", True)
        bh_layout = QVBoxLayout()
        bh_layout.setSpacing(2)
        bh_layout.setContentsMargins(4, 4, 4, 4)

        bh_step_layout = QHBoxLayout()
        bh_step_layout.addWidget(QLabel("扰动步长:"))
        self.bh_step_size = QDoubleSpinBox()
        self.bh_step_size.setRange(0.01, 5.0)
        self.bh_step_size.setValue(0.5)
        self.bh_step_size.setDecimals(2)
        self.bh_step_size.setSingleStep(0.1)
        bh_step_layout.addWidget(self.bh_step_size)
        bh_layout.addLayout(bh_step_layout)

        bh_temp_layout = QHBoxLayout()
        bh_temp_layout.addWidget(QLabel("MC温度:"))
        self.bh_temperature = QDoubleSpinBox()
        self.bh_temperature.setRange(0.01, 100.0)
        self.bh_temperature.setValue(1.0)
        self.bh_temperature.setDecimals(2)
        self.bh_temperature.setSingleStep(0.1)
        bh_temp_layout.addWidget(self.bh_temperature)
        bh_layout.addLayout(bh_temp_layout)

        bh_iter_layout = QHBoxLayout()
        bh_iter_layout.addWidget(QLabel("最大迭代:"))
        self.bh_max_iter = QSpinBox()
        self.bh_max_iter.setRange(10, 500)
        self.bh_max_iter.setValue(100)
        bh_iter_layout.addWidget(self.bh_max_iter)
        bh_layout.addLayout(bh_iter_layout)

        self.bh_confirm_btn = QPushButton("确认参数")
        self.bh_confirm_btn.setObjectName("applyRangeBtn")
        self.bh_confirm_btn.setFixedHeight(22)
        bh_layout.addWidget(self.bh_confirm_btn)

        self.bh_group.setLayout(bh_layout)
        layout.addWidget(self.bh_group)

        # Metropolis MC参数
        self.mc_group = QGroupBox("Metropolis MC参数")
        self.mc_group.setProperty("flat", True)
        mc_layout = QVBoxLayout()
        mc_layout.setSpacing(2)
        mc_layout.setContentsMargins(4, 4, 4, 4)

        mc_step_layout = QHBoxLayout()
        mc_step_layout.addWidget(QLabel("扰动步长:"))
        self.mc_step_size = QDoubleSpinBox()
        self.mc_step_size.setRange(0.01, 5.0)
        self.mc_step_size.setValue(0.5)
        self.mc_step_size.setDecimals(2)
        self.mc_step_size.setSingleStep(0.1)
        mc_step_layout.addWidget(self.mc_step_size)
        mc_layout.addLayout(mc_step_layout)

        mc_temp_layout = QHBoxLayout()
        mc_temp_layout.addWidget(QLabel("温度kT:"))
        self.mc_temperature = QDoubleSpinBox()
        self.mc_temperature.setRange(0.01, 100.0)
        self.mc_temperature.setValue(1.0)
        self.mc_temperature.setDecimals(2)
        self.mc_temperature.setSingleStep(0.1)
        mc_temp_layout.addWidget(self.mc_temperature)
        mc_layout.addLayout(mc_temp_layout)

        mc_iter_layout = QHBoxLayout()
        mc_iter_layout.addWidget(QLabel("最大迭代:"))
        self.mc_max_iter = QSpinBox()
        self.mc_max_iter.setRange(10, 1000)
        self.mc_max_iter.setValue(200)
        mc_iter_layout.addWidget(self.mc_max_iter)
        mc_layout.addLayout(mc_iter_layout)

        self.mc_confirm_btn = QPushButton("确认参数")
        self.mc_confirm_btn.setObjectName("applyRangeBtn")
        self.mc_confirm_btn.setFixedHeight(22)
        mc_layout.addWidget(self.mc_confirm_btn)

        self.mc_group.setLayout(mc_layout)
        layout.addWidget(self.mc_group)

        # SSW方法参数
        self.sSW_group = QGroupBox("SSW方法参数")
        self.sSW_group.setProperty("flat", True)
        sSW_layout = QVBoxLayout()
        sSW_layout.setSpacing(2)
        sSW_layout.setContentsMargins(4, 4, 4, 4)

        sSW_step_layout = QHBoxLayout()
        sSW_step_layout.addWidget(QLabel("扰动步长:"))
        self.sSW_step_size = QDoubleSpinBox()
        self.sSW_step_size.setRange(0.01, 5.0)
        self.sSW_step_size.setValue(0.5)
        self.sSW_step_size.setDecimals(2)
        self.sSW_step_size.setSingleStep(0.1)
        sSW_step_layout.addWidget(self.sSW_step_size)
        sSW_layout.addLayout(sSW_step_layout)

        sSW_gh_layout = QHBoxLayout()
        sSW_gh_layout.addWidget(QLabel("高斯高度:"))
        self.sSW_gaussian_height = QDoubleSpinBox()
        self.sSW_gaussian_height.setRange(0.1, 50.0)
        self.sSW_gaussian_height.setValue(5.0)
        self.sSW_gaussian_height.setDecimals(1)
        self.sSW_gaussian_height.setSingleStep(0.5)
        sSW_gh_layout.addWidget(self.sSW_gaussian_height)
        sSW_layout.addLayout(sSW_gh_layout)

        sSW_gw_layout = QHBoxLayout()
        sSW_gw_layout.addWidget(QLabel("高斯宽度:"))
        self.sSW_gaussian_width = QDoubleSpinBox()
        self.sSW_gaussian_width.setRange(0.1, 5.0)
        self.sSW_gaussian_width.setValue(0.5)
        self.sSW_gaussian_width.setDecimals(1)
        self.sSW_gaussian_width.setSingleStep(0.1)
        sSW_gw_layout.addWidget(self.sSW_gaussian_width)
        sSW_layout.addLayout(sSW_gw_layout)

        sSW_temp_layout = QHBoxLayout()
        sSW_temp_layout.addWidget(QLabel("MC温度:"))
        self.sSW_temperature = QDoubleSpinBox()
        self.sSW_temperature.setRange(0.01, 100.0)
        self.sSW_temperature.setValue(1.0)
        self.sSW_temperature.setDecimals(2)
        self.sSW_temperature.setSingleStep(0.1)
        sSW_temp_layout.addWidget(self.sSW_temperature)
        sSW_layout.addLayout(sSW_temp_layout)

        sSW_iter_layout = QHBoxLayout()
        sSW_iter_layout.addWidget(QLabel("最大迭代:"))
        self.sSW_max_iter = QSpinBox()
        self.sSW_max_iter.setRange(10, 500)
        self.sSW_max_iter.setValue(100)
        sSW_iter_layout.addWidget(self.sSW_max_iter)
        sSW_layout.addLayout(sSW_iter_layout)

        self.sSW_confirm_btn = QPushButton("确认参数")
        self.sSW_confirm_btn.setObjectName("applyRangeBtn")
        self.sSW_confirm_btn.setFixedHeight(22)
        sSW_layout.addWidget(self.sSW_confirm_btn)

        self.sSW_group.setLayout(sSW_layout)
        layout.addWidget(self.sSW_group)

        # 元动力学参数
        self.meta_group = QGroupBox("元动力学参数")
        self.meta_group.setProperty("flat", True)
        meta_layout = QVBoxLayout()
        meta_layout.setSpacing(2)
        meta_layout.setContentsMargins(4, 4, 4, 4)

        meta_gh_layout = QHBoxLayout()
        meta_gh_layout.addWidget(QLabel("高斯高度:"))
        self.meta_gaussian_height = QDoubleSpinBox()
        self.meta_gaussian_height.setRange(0.01, 10.0)
        self.meta_gaussian_height.setValue(1.0)
        self.meta_gaussian_height.setDecimals(2)
        self.meta_gaussian_height.setSingleStep(0.1)
        meta_gh_layout.addWidget(self.meta_gaussian_height)
        meta_layout.addLayout(meta_gh_layout)

        meta_gw_layout = QHBoxLayout()
        meta_gw_layout.addWidget(QLabel("高斯宽度:"))
        self.meta_gaussian_width = QDoubleSpinBox()
        self.meta_gaussian_width.setRange(0.05, 5.0)
        self.meta_gaussian_width.setValue(0.3)
        self.meta_gaussian_width.setDecimals(2)
        self.meta_gaussian_width.setSingleStep(0.05)
        meta_gw_layout.addWidget(self.meta_gaussian_width)
        meta_layout.addLayout(meta_gw_layout)

        meta_iter_layout = QHBoxLayout()
        meta_iter_layout.addWidget(QLabel("最大迭代:"))
        self.meta_max_iter = QSpinBox()
        self.meta_max_iter.setRange(10, 1000)
        self.meta_max_iter.setValue(200)
        meta_iter_layout.addWidget(self.meta_max_iter)
        meta_layout.addLayout(meta_iter_layout)

        self.meta_confirm_btn = QPushButton("确认参数")
        self.meta_confirm_btn.setObjectName("applyRangeBtn")
        self.meta_confirm_btn.setFixedHeight(22)
        meta_layout.addWidget(self.meta_confirm_btn)

        self.meta_group.setLayout(meta_layout)
        layout.addWidget(self.meta_group)

        # 极小点跳跃参数
        self.mh_group = QGroupBox("极小点跳跃参数")
        self.mh_group.setProperty("flat", True)
        mh_layout = QVBoxLayout()
        mh_layout.setSpacing(2)
        mh_layout.setContentsMargins(4, 4, 4, 4)

        mh_kinetic_layout = QHBoxLayout()
        mh_kinetic_layout.addWidget(QLabel("初始动能:"))
        self.mh_kinetic = QDoubleSpinBox()
        self.mh_kinetic.setRange(0.01, 10.0)
        self.mh_kinetic.setValue(0.5)
        self.mh_kinetic.setDecimals(2)
        self.mh_kinetic.setSingleStep(0.1)
        mh_kinetic_layout.addWidget(self.mh_kinetic)
        mh_layout.addLayout(mh_kinetic_layout)

        mh_iter_layout = QHBoxLayout()
        mh_iter_layout.addWidget(QLabel("最大迭代:"))
        self.mh_max_iter = QSpinBox()
        self.mh_max_iter.setRange(10, 500)
        self.mh_max_iter.setValue(100)
        mh_iter_layout.addWidget(self.mh_max_iter)
        mh_layout.addLayout(mh_iter_layout)

        self.mh_confirm_btn = QPushButton("确认参数")
        self.mh_confirm_btn.setObjectName("applyRangeBtn")
        self.mh_confirm_btn.setFixedHeight(22)
        mh_layout.addWidget(self.mh_confirm_btn)

        self.mh_group.setLayout(mh_layout)
        layout.addWidget(self.mh_group)

        # 遗传算法参数
        self.ga_group = QGroupBox("遗传算法参数")
        self.ga_group.setProperty("flat", True)
        ga_layout = QVBoxLayout()
        ga_layout.setSpacing(2)
        ga_layout.setContentsMargins(4, 4, 4, 4)

        ga_pop_layout = QHBoxLayout()
        ga_pop_layout.addWidget(QLabel("种群大小:"))
        self.ga_pop_size = QSpinBox()
        self.ga_pop_size.setRange(5, 100)
        self.ga_pop_size.setValue(20)
        ga_pop_layout.addWidget(self.ga_pop_size)
        ga_layout.addLayout(ga_pop_layout)

        ga_mut_layout = QHBoxLayout()
        ga_mut_layout.addWidget(QLabel("变异率:"))
        self.ga_mutation_rate = QDoubleSpinBox()
        self.ga_mutation_rate.setRange(0.0, 1.0)
        self.ga_mutation_rate.setValue(0.3)
        self.ga_mutation_rate.setDecimals(2)
        self.ga_mutation_rate.setSingleStep(0.05)
        ga_mut_layout.addWidget(self.ga_mutation_rate)
        ga_layout.addLayout(ga_mut_layout)

        ga_iter_layout = QHBoxLayout()
        ga_iter_layout.addWidget(QLabel("最大迭代:"))
        self.ga_max_iter = QSpinBox()
        self.ga_max_iter.setRange(10, 500)
        self.ga_max_iter.setValue(100)
        ga_iter_layout.addWidget(self.ga_max_iter)
        ga_layout.addLayout(ga_iter_layout)

        ga_range_layout = QHBoxLayout()
        ga_range_layout.addWidget(QLabel("搜索范围:"))
        self.ga_search_range = QDoubleSpinBox()
        self.ga_search_range.setRange(0.5, 10.0)
        self.ga_search_range.setValue(2.0)
        self.ga_search_range.setDecimals(1)
        self.ga_search_range.setSingleStep(0.5)
        ga_range_layout.addWidget(self.ga_search_range)
        ga_layout.addLayout(ga_range_layout)

        self.ga_confirm_btn = QPushButton("确认参数")
        self.ga_confirm_btn.setObjectName("applyRangeBtn")
        self.ga_confirm_btn.setFixedHeight(22)
        ga_layout.addWidget(self.ga_confirm_btn)

        self.ga_group.setLayout(ga_layout)
        layout.addWidget(self.ga_group)

        # 粒子群优化参数
        self.pso_group = QGroupBox("粒子群优化参数")
        self.pso_group.setProperty("flat", True)
        pso_layout = QVBoxLayout()
        pso_layout.setSpacing(2)
        pso_layout.setContentsMargins(4, 4, 4, 4)

        pso_np_layout = QHBoxLayout()
        pso_np_layout.addWidget(QLabel("粒子数:"))
        self.pso_n_particles = QSpinBox()
        self.pso_n_particles.setRange(5, 100)
        self.pso_n_particles.setValue(20)
        pso_np_layout.addWidget(self.pso_n_particles)
        pso_layout.addLayout(pso_np_layout)

        pso_w_layout = QHBoxLayout()
        pso_w_layout.addWidget(QLabel("惯性权重:"))
        self.pso_w = QDoubleSpinBox()
        self.pso_w.setRange(0.0, 2.0)
        self.pso_w.setValue(0.7)
        self.pso_w.setDecimals(1)
        self.pso_w.setSingleStep(0.1)
        pso_w_layout.addWidget(self.pso_w)
        pso_layout.addLayout(pso_w_layout)

        pso_c1_layout = QHBoxLayout()
        pso_c1_layout.addWidget(QLabel("认知系数:"))
        self.pso_c1 = QDoubleSpinBox()
        self.pso_c1.setRange(0.0, 3.0)
        self.pso_c1.setValue(1.5)
        self.pso_c1.setDecimals(1)
        self.pso_c1.setSingleStep(0.1)
        pso_c1_layout.addWidget(self.pso_c1)
        pso_layout.addLayout(pso_c1_layout)

        pso_c2_layout = QHBoxLayout()
        pso_c2_layout.addWidget(QLabel("社会系数:"))
        self.pso_c2 = QDoubleSpinBox()
        self.pso_c2.setRange(0.0, 3.0)
        self.pso_c2.setValue(1.5)
        self.pso_c2.setDecimals(1)
        self.pso_c2.setSingleStep(0.1)
        pso_c2_layout.addWidget(self.pso_c2)
        pso_layout.addLayout(pso_c2_layout)

        pso_iter_layout = QHBoxLayout()
        pso_iter_layout.addWidget(QLabel("最大迭代:"))
        self.pso_max_iter = QSpinBox()
        self.pso_max_iter.setRange(10, 500)
        self.pso_max_iter.setValue(100)
        pso_iter_layout.addWidget(self.pso_max_iter)
        pso_layout.addLayout(pso_iter_layout)

        self.pso_confirm_btn = QPushButton("确认参数")
        self.pso_confirm_btn.setObjectName("applyRangeBtn")
        self.pso_confirm_btn.setFixedHeight(22)
        pso_layout.addWidget(self.pso_confirm_btn)

        self.pso_group.setLayout(pso_layout)
        layout.addWidget(self.pso_group)

        # 人工蜂群参数
        self.abc_group = QGroupBox("人工蜂群参数")
        self.abc_group.setProperty("flat", True)
        abc_layout = QVBoxLayout()
        abc_layout.setSpacing(2)
        abc_layout.setContentsMargins(4, 4, 4, 4)

        abc_nb_layout = QHBoxLayout()
        abc_nb_layout.addWidget(QLabel("蜜蜂数:"))
        self.abc_n_bees = QSpinBox()
        self.abc_n_bees.setRange(5, 100)
        self.abc_n_bees.setValue(20)
        abc_nb_layout.addWidget(self.abc_n_bees)
        abc_layout.addLayout(abc_nb_layout)

        abc_limit_layout = QHBoxLayout()
        abc_limit_layout.addWidget(QLabel("放弃阈值:"))
        self.abc_limit = QSpinBox()
        self.abc_limit.setRange(10, 200)
        self.abc_limit.setValue(50)
        abc_limit_layout.addWidget(self.abc_limit)
        abc_layout.addLayout(abc_limit_layout)

        abc_iter_layout = QHBoxLayout()
        abc_iter_layout.addWidget(QLabel("最大迭代:"))
        self.abc_max_iter = QSpinBox()
        self.abc_max_iter.setRange(10, 500)
        self.abc_max_iter.setValue(100)
        abc_iter_layout.addWidget(self.abc_max_iter)
        abc_layout.addLayout(abc_iter_layout)

        abc_sr_layout = QHBoxLayout()
        abc_sr_layout.addWidget(QLabel("搜索范围:"))
        self.abc_search_range = QDoubleSpinBox()
        self.abc_search_range.setRange(0.5, 10.0)
        self.abc_search_range.setValue(2.0)
        self.abc_search_range.setDecimals(1)
        self.abc_search_range.setSingleStep(0.5)
        abc_sr_layout.addWidget(self.abc_search_range)
        abc_layout.addLayout(abc_sr_layout)

        self.abc_confirm_btn = QPushButton("确认参数")
        self.abc_confirm_btn.setObjectName("applyRangeBtn")
        self.abc_confirm_btn.setFixedHeight(22)
        abc_layout.addWidget(self.abc_confirm_btn)

        self.abc_group.setLayout(abc_layout)
        layout.addWidget(self.abc_group)

        # 伞形采样参数
        self.us_group = QGroupBox("伞形采样参数")
        self.us_group.setProperty("flat", True)
        us_layout = QVBoxLayout()
        us_layout.setSpacing(2)
        us_layout.setContentsMargins(4, 4, 4, 4)

        us_nw_layout = QHBoxLayout()
        us_nw_layout.addWidget(QLabel("窗口数:"))
        self.us_n_windows = QSpinBox()
        self.us_n_windows.setRange(2, 20)
        self.us_n_windows.setValue(5)
        us_nw_layout.addWidget(self.us_n_windows)
        us_layout.addLayout(us_nw_layout)

        us_k_layout = QHBoxLayout()
        us_k_layout.addWidget(QLabel("弹簧常数 k:"))
        self.us_spring_k = QDoubleSpinBox()
        self.us_spring_k.setRange(0.1, 100.0)
        self.us_spring_k.setValue(10.0)
        self.us_spring_k.setDecimals(1)
        self.us_spring_k.setSingleStep(1.0)
        us_k_layout.addWidget(self.us_spring_k)
        us_layout.addLayout(us_k_layout)

        us_ss_layout = QHBoxLayout()
        us_ss_layout.addWidget(QLabel("MC步长:"))
        self.us_mc_step_size = QDoubleSpinBox()
        self.us_mc_step_size.setRange(0.01, 2.0)
        self.us_mc_step_size.setValue(0.3)
        self.us_mc_step_size.setDecimals(2)
        self.us_mc_step_size.setSingleStep(0.01)
        us_ss_layout.addWidget(self.us_mc_step_size)
        us_layout.addLayout(us_ss_layout)

        us_temp_layout = QHBoxLayout()
        us_temp_layout.addWidget(QLabel("MC温度:"))
        self.us_mc_temperature = QDoubleSpinBox()
        self.us_mc_temperature.setRange(0.01, 1000.0)
        self.us_mc_temperature.setValue(50.0)
        self.us_mc_temperature.setDecimals(2)
        self.us_mc_temperature.setSingleStep(0.1)
        us_temp_layout.addWidget(self.us_mc_temperature)
        us_layout.addLayout(us_temp_layout)

        us_spw_layout = QHBoxLayout()
        us_spw_layout.addWidget(QLabel("每窗口步数:"))
        self.us_steps_per_window = QSpinBox()
        self.us_steps_per_window.setRange(5, 200)
        self.us_steps_per_window.setValue(20)
        us_spw_layout.addWidget(self.us_steps_per_window)
        us_layout.addLayout(us_spw_layout)

        us_iter_layout = QHBoxLayout()
        us_iter_layout.addWidget(QLabel("最大迭代:"))
        self.us_max_iter = QSpinBox()
        self.us_max_iter.setRange(10, 2000)
        self.us_max_iter.setValue(100)
        us_iter_layout.addWidget(self.us_max_iter)
        us_layout.addLayout(us_iter_layout)

        self.us_confirm_btn = QPushButton("确认参数")
        self.us_confirm_btn.setObjectName("applyRangeBtn")
        self.us_confirm_btn.setFixedHeight(22)
        us_layout.addWidget(self.us_confirm_btn)

        self.us_group.setLayout(us_layout)
        layout.addWidget(self.us_group)

        # 自适应偏置力参数
        self.abf_group = QGroupBox("自适应偏置力参数")
        self.abf_group.setProperty("flat", True)
        abf_layout = QVBoxLayout()
        abf_layout.setSpacing(2)
        abf_layout.setContentsMargins(4, 4, 4, 4)

        abf_nb_layout = QHBoxLayout()
        abf_nb_layout.addWidget(QLabel("分箱数:"))
        self.abf_n_bins = QSpinBox()
        self.abf_n_bins.setRange(5, 100)
        self.abf_n_bins.setValue(20)
        abf_nb_layout.addWidget(self.abf_n_bins)
        abf_layout.addLayout(abf_nb_layout)

        abf_ss_layout = QHBoxLayout()
        abf_ss_layout.addWidget(QLabel("MC步长:"))
        self.abf_mc_step_size = QDoubleSpinBox()
        self.abf_mc_step_size.setRange(0.01, 2.0)
        self.abf_mc_step_size.setValue(0.3)
        self.abf_mc_step_size.setDecimals(2)
        self.abf_mc_step_size.setSingleStep(0.01)
        abf_ss_layout.addWidget(self.abf_mc_step_size)
        abf_layout.addLayout(abf_ss_layout)

        abf_temp_layout = QHBoxLayout()
        abf_temp_layout.addWidget(QLabel("MC温度:"))
        self.abf_mc_temperature = QDoubleSpinBox()
        self.abf_mc_temperature.setRange(0.01, 1000.0)
        self.abf_mc_temperature.setValue(50.0)
        self.abf_mc_temperature.setDecimals(2)
        self.abf_mc_temperature.setSingleStep(0.1)
        abf_temp_layout.addWidget(self.abf_mc_temperature)
        abf_layout.addLayout(abf_temp_layout)

        abf_iter_layout = QHBoxLayout()
        abf_iter_layout.addWidget(QLabel("最大迭代:"))
        self.abf_max_iter = QSpinBox()
        self.abf_max_iter.setRange(10, 2000)
        self.abf_max_iter.setValue(200)
        abf_iter_layout.addWidget(self.abf_max_iter)
        abf_layout.addLayout(abf_iter_layout)

        self.abf_confirm_btn = QPushButton("确认参数")
        self.abf_confirm_btn.setObjectName("applyRangeBtn")
        self.abf_confirm_btn.setFixedHeight(22)
        abf_layout.addWidget(self.abf_confirm_btn)

        self.abf_group.setLayout(abf_layout)
        layout.addWidget(self.abf_group)

        # CBD方法参数
        self.cbd_group = QGroupBox("CBD方法参数")
        self.cbd_group.setProperty("flat", True)
        cbd_layout = QVBoxLayout()
        cbd_layout.setSpacing(2)
        cbd_layout.setContentsMargins(4, 4, 4, 4)

        cbd_dr_layout = QHBoxLayout()
        cbd_dr_layout.addWidget(QLabel("双子间距 ΔR:"))
        self.cbd_delta_r = QDoubleSpinBox()
        self.cbd_delta_r.setRange(0.001, 0.1)
        self.cbd_delta_r.setValue(0.01)
        self.cbd_delta_r.setDecimals(3)
        self.cbd_delta_r.setSingleStep(0.001)
        cbd_dr_layout.addWidget(self.cbd_delta_r)
        cbd_layout.addLayout(cbd_dr_layout)

        cbd_thr_layout = QHBoxLayout()
        cbd_thr_layout.addWidget(QLabel("收敛阈值:"))
        self.cbd_threshold = QDoubleSpinBox()
        self.cbd_threshold.setRange(1e-4, 1e-1)
        self.cbd_threshold.setValue(1e-3)
        self.cbd_threshold.setDecimals(4)
        self.cbd_threshold.setSingleStep(1e-4)
        cbd_thr_layout.addWidget(self.cbd_threshold)
        cbd_layout.addLayout(cbd_thr_layout)

        cbd_step_layout = QHBoxLayout()
        cbd_step_layout.addWidget(QLabel("平动步长:"))
        self.cbd_trans_step = QDoubleSpinBox()
        self.cbd_trans_step.setRange(0.01, 1.0)
        self.cbd_trans_step.setValue(0.1)
        self.cbd_trans_step.setDecimals(2)
        self.cbd_trans_step.setSingleStep(0.01)
        cbd_step_layout.addWidget(self.cbd_trans_step)
        cbd_layout.addLayout(cbd_step_layout)

        cbd_iter_layout = QHBoxLayout()
        cbd_iter_layout.addWidget(QLabel("最大迭代:"))
        self.cbd_max_iter = QSpinBox()
        self.cbd_max_iter.setRange(10, 500)
        self.cbd_max_iter.setValue(100)
        cbd_iter_layout.addWidget(self.cbd_max_iter)
        cbd_layout.addLayout(cbd_iter_layout)

        self.cbd_confirm_btn = QPushButton("确认参数")
        self.cbd_confirm_btn.setObjectName("applyRangeBtn")
        self.cbd_confirm_btn.setFixedHeight(22)
        cbd_layout.addWidget(self.cbd_confirm_btn)

        self.cbd_group.setLayout(cbd_layout)
        layout.addWidget(self.cbd_group)

        # DESW方法参数
        self.desw_group = QGroupBox("DESW方法参数")
        self.desw_group.setProperty("flat", True)
        desw_layout = QVBoxLayout()
        desw_layout.setSpacing(2)
        desw_layout.setContentsMargins(4, 4, 4, 4)

        desw_ss_layout = QHBoxLayout()
        desw_ss_layout.addWidget(QLabel("扰动步长:"))
        self.desw_step_size = QDoubleSpinBox()
        self.desw_step_size.setRange(0.01, 5.0)
        self.desw_step_size.setValue(0.5)
        self.desw_step_size.setDecimals(2)
        self.desw_step_size.setSingleStep(0.1)
        desw_ss_layout.addWidget(self.desw_step_size)
        desw_layout.addLayout(desw_ss_layout)

        desw_gh_layout = QHBoxLayout()
        desw_gh_layout.addWidget(QLabel("高斯高度:"))
        self.desw_gaussian_height = QDoubleSpinBox()
        self.desw_gaussian_height.setRange(0.1, 50.0)
        self.desw_gaussian_height.setValue(5.0)
        self.desw_gaussian_height.setDecimals(1)
        self.desw_gaussian_height.setSingleStep(0.5)
        desw_gh_layout.addWidget(self.desw_gaussian_height)
        desw_layout.addLayout(desw_gh_layout)

        desw_gw_layout = QHBoxLayout()
        desw_gw_layout.addWidget(QLabel("高斯宽度:"))
        self.desw_gaussian_width = QDoubleSpinBox()
        self.desw_gaussian_width.setRange(0.1, 5.0)
        self.desw_gaussian_width.setValue(0.5)
        self.desw_gaussian_width.setDecimals(1)
        self.desw_gaussian_width.setSingleStep(0.1)
        desw_gw_layout.addWidget(self.desw_gaussian_width)
        desw_layout.addLayout(desw_gw_layout)

        desw_temp_layout = QHBoxLayout()
        desw_temp_layout.addWidget(QLabel("MC温度:"))
        self.desw_temperature = QDoubleSpinBox()
        self.desw_temperature.setRange(0.01, 100.0)
        self.desw_temperature.setValue(1.0)
        self.desw_temperature.setDecimals(2)
        self.desw_temperature.setSingleStep(0.1)
        desw_temp_layout.addWidget(self.desw_temperature)
        desw_layout.addLayout(desw_temp_layout)

        desw_iter_layout = QHBoxLayout()
        desw_iter_layout.addWidget(QLabel("最大迭代:"))
        self.desw_max_iter = QSpinBox()
        self.desw_max_iter.setRange(10, 500)
        self.desw_max_iter.setValue(100)
        desw_iter_layout.addWidget(self.desw_max_iter)
        desw_layout.addLayout(desw_iter_layout)

        self.desw_confirm_btn = QPushButton("确认参数")
        self.desw_confirm_btn.setObjectName("applyRangeBtn")
        self.desw_confirm_btn.setFixedHeight(22)
        desw_layout.addWidget(self.desw_confirm_btn)

        self.desw_group.setLayout(desw_layout)
        layout.addWidget(self.desw_group)

        # ---- 初始点设置区 ----
        self.init_group = QGroupBox("初始点设置")
        self.init_group.setProperty("flat", True)
        init_layout = QVBoxLayout()
        init_layout.setSpacing(2)
        init_layout.setContentsMargins(4, 4, 4, 4)

        self.init_hint = QLabel("请在势能面上点击设置初始点")
        self.init_hint.setWordWrap(True)
        self.init_hint.setStyleSheet("color: #00d4ff; font-weight: bold;")
        init_layout.addWidget(self.init_hint)

        coord1_layout = QHBoxLayout()
        coord1_layout.addWidget(QLabel("初始点 x:"))
        self.init_x1 = QLineEdit()
        self.init_x1.setPlaceholderText("x")
        coord1_layout.addWidget(self.init_x1)
        coord1_layout.addWidget(QLabel("y:"))
        self.init_y1 = QLineEdit()
        self.init_y1.setPlaceholderText("y")
        coord1_layout.addWidget(self.init_y1)
        init_layout.addLayout(coord1_layout)

        self.init_point2_label = QLabel("终态坐标:")
        self.init_point2_label.setVisible(False)
        init_layout.addWidget(self.init_point2_label)

        self.coord2_layout_widget = QWidget()
        coord2_l = QHBoxLayout(self.coord2_layout_widget)
        coord2_l.setContentsMargins(0, 0, 0, 0)
        coord2_l.setSpacing(2)
        self.init_x2 = QLineEdit()
        self.init_x2.setPlaceholderText("x")
        self.init_y2 = QLineEdit()
        self.init_y2.setPlaceholderText("y")
        coord2_l.addWidget(QLabel("终态 x:"))
        coord2_l.addWidget(self.init_x2)
        coord2_l.addWidget(QLabel("y:"))
        coord2_l.addWidget(self.init_y2)
        self.coord2_layout_widget.setVisible(False)
        init_layout.addWidget(self.coord2_layout_widget)

        init_btn_layout = QHBoxLayout()
        self.set_point_btn = QPushButton("设置")
        self.set_point_btn.setObjectName("setPointBtn")
        self.finish_point_btn = QPushButton("完成")
        self.finish_point_btn.setObjectName("finishPointBtn")
        self.finish_point_btn.setEnabled(False)
        init_btn_layout.addWidget(self.set_point_btn)
        init_btn_layout.addWidget(self.finish_point_btn)
        init_layout.addLayout(init_btn_layout)

        self.init_group.setLayout(init_layout)
        layout.addWidget(self.init_group)

        # ---- 播放控制区 ----
        play_group = QGroupBox("播放控制")
        play_group.setProperty("flat", True)
        play_layout = QVBoxLayout()
        play_layout.setSpacing(2)
        play_layout.setContentsMargins(4, 4, 4, 4)

        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("开始")
        self.play_btn.setObjectName("playBtn")
        self.step_btn = QPushButton("单步")
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setObjectName("resetBtn")
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.step_btn)
        btn_layout.addWidget(self.reset_btn)
        play_layout.addLayout(btn_layout)

        speed_layout = QHBoxLayout()
        self.speed_slow = QRadioButton("慢速")
        self.speed_mid = QRadioButton("中速")
        self.speed_fast = QRadioButton("快速")
        self.speed_mid.setChecked(True)
        self.speed_group = QButtonGroup()
        self.speed_group.addButton(self.speed_slow)
        self.speed_group.addButton(self.speed_mid)
        self.speed_group.addButton(self.speed_fast)
        speed_layout.addWidget(self.speed_slow)
        speed_layout.addWidget(self.speed_mid)
        speed_layout.addWidget(self.speed_fast)
        play_layout.addLayout(speed_layout)

        self.back_btn = QPushButton("回退一步")
        play_layout.addWidget(self.back_btn)

        play_group.setLayout(play_layout)
        layout.addWidget(play_group)

        # ---- 方法对比区 ----
        compare_group = QGroupBox("方法对比")
        compare_group.setProperty("flat", True)
        compare_layout = QVBoxLayout()
        compare_layout.setSpacing(2)
        compare_layout.setContentsMargins(4, 4, 4, 4)
        self.compare_check = QCheckBox("启用方法对比")
        compare_layout.addWidget(self.compare_check)

        cmp_algo_layout = QHBoxLayout()
        self.compare_algo_label = QLabel("对比:")
        self.compare_algo_combo = QComboBox()
        self.compare_algo_combo.addItems([
            "牛顿法(极小值)", "牛顿法(鞍点)", "Dimer方法",
            "NEB方法", "CI-NEB方法", "最速下降法", "盆地跳跃法", "Metropolis MC",
            "SSW方法", "元动力学", "极小点跳跃", "遗传算法", "粒子群优化",
            "人工蜂群", "伞形采样", "自适应偏置力", "CBD方法", "DESW方法"
        ])
        cmp_algo_layout.addWidget(self.compare_algo_label)
        cmp_algo_layout.addWidget(self.compare_algo_combo)
        self.compare_edit_params_btn = QPushButton("编辑参数")
        self.compare_edit_params_btn.setToolTip("切换参数面板到对比方法，设置参数后再点此按钮返回")
        self.compare_edit_params_btn.setVisible(False)
        cmp_algo_layout.addWidget(self.compare_edit_params_btn)
        self.compare_algo_label.setVisible(False)
        self.compare_algo_combo.setVisible(False)
        compare_layout.addLayout(cmp_algo_layout)

        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)

        # ---- 导出区 ----
        export_group = QGroupBox("导出")
        export_group.setProperty("flat", True)
        export_layout = QVBoxLayout()
        export_layout.setSpacing(2)
        export_layout.setContentsMargins(4, 4, 4, 4)
        self.export_csv_btn = QPushButton("导出轨迹数据(CSV)")
        self.export_png_btn = QPushButton("导出势能面图片(PNG)")
        self.export_gif_btn = QPushButton("导出搜索过程GIF")
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_png_btn)
        export_layout.addWidget(self.export_gif_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # ---- 说明按钮（单独放置） ----
        self.help_btn = QPushButton("使用说明")
        self.help_btn.setObjectName("helpBtn")
        layout.addWidget(self.help_btn)

        layout.addStretch()

        # 初始状态：显示牛顿法参数
        self._update_algo_params()

        # 连接信号
        self.algo_combo.currentIndexChanged.connect(self._update_algo_params)
        self.pes_combo.currentIndexChanged.connect(self._update_pes_ui)
        self.compare_check.stateChanged.connect(self._update_compare_ui)
        self.contour_check.stateChanged.connect(self._update_mode_check)
        self.surface3d_check.stateChanged.connect(self._update_mode_check)

    def _update_pes_ui(self):
        """更新势能面选择UI"""
        is_custom = self.pes_combo.currentText() == "自定义"
        self.custom_expr_label.setVisible(is_custom)
        self.custom_expr.setVisible(is_custom)
        self.custom_expr_confirm_btn.setVisible(is_custom)

    def _update_mode_check(self):
        """确保渲染模式互斥"""
        sender = self.sender()
        if sender == self.contour_check and self.contour_check.isChecked():
            self.surface3d_check.setChecked(False)
        elif sender == self.surface3d_check and self.surface3d_check.isChecked():
            self.contour_check.setChecked(False)
        # 至少选一个
        if not self.contour_check.isChecked() and not self.surface3d_check.isChecked():
            self.contour_check.setChecked(True)

    def _update_algo_params(self):
        """根据算法选择动态显示参数控件"""
        algo = self.algo_combo.currentText()
        is_newton = algo in ("牛顿法(极小值)", "牛顿法(鞍点)")
        is_dimer = algo == "Dimer方法"
        is_neb = algo in ("NEB方法", "CI-NEB方法")

        self.newton_group.setVisible(is_newton)
        self.dimer_group.setVisible(is_dimer)
        self.neb_group.setVisible(is_neb)
        self.sd_group.setVisible(algo == "最速下降法")
        self.bh_group.setVisible(algo == "盆地跳跃法")
        self.mc_group.setVisible(algo == "Metropolis MC")
        self.sSW_group.setVisible(algo == "SSW方法")
        self.meta_group.setVisible(algo == "元动力学")
        self.mh_group.setVisible(algo == "极小点跳跃")
        self.ga_group.setVisible(algo == "遗传算法")
        self.pso_group.setVisible(algo == "粒子群优化")
        self.abc_group.setVisible(algo == "人工蜂群")
        self.us_group.setVisible(algo == "伞形采样")
        self.abf_group.setVisible(algo == "自适应偏置力")
        self.cbd_group.setVisible(algo == "CBD方法")
        self.desw_group.setVisible(algo == "DESW方法")

        # 更新初始点提示
        if is_neb:
            self.init_hint.setText("请依次点击设置初态和终态")
            self.init_point2_label.setVisible(True)
            self.coord2_layout_widget.setVisible(True)
        elif algo in ("伞形采样", "自适应偏置力", "DESW方法"):
            self.init_hint.setText("请依次点击设置起点和终点（反应坐标）")
            self.init_point2_label.setVisible(True)
            self.coord2_layout_widget.setVisible(True)
        else:
            self.init_hint.setText("请在势能面上点击设置初始点")
            self.init_point2_label.setVisible(False)
            self.coord2_layout_widget.setVisible(False)

    def _update_compare_ui(self):
        """更新方法对比UI"""
        enabled = self.compare_check.isChecked()
        self.compare_algo_label.setVisible(enabled)
        self.compare_algo_combo.setVisible(enabled)
        self.compare_edit_params_btn.setVisible(enabled)

    def get_speed_ms(self):
        """获取当前播放速度（毫秒）"""
        if self.speed_slow.isChecked():
            return 500
        elif self.speed_fast.isChecked():
            return 50
        else:
            return 200

    def is_contour_mode(self):
        """是否为等高线模式"""
        return self.contour_check.isChecked()

    def is_3d_mode(self):
        """是否为3D模式"""
        return self.surface3d_check.isChecked()

    def get_algo_type(self):
        """获取当前算法类型"""
        return self.algo_combo.currentText()


# ======================== 下方信息面板 ========================

class InfoPanel(QWidget):
    """下方信息面板：实时物理量、原理说明、关键术语/误区、结果统计并排显示"""

    # 基准高度和基准字号
    BASE_HEIGHT = 200
    BASE_FONT_SIZE = 9
    BASE_TERMS_FONT_SIZE = 10  # 关键术语/误区基础字号（稍大）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # ---- 实时物理量区 ----
        physics_group = QGroupBox("实时物理量")
        physics_group.setProperty("flat", True)
        physics_group.setObjectName("infoPhysics")
        physics_layout = QVBoxLayout()
        physics_layout.setSpacing(2)
        physics_layout.setContentsMargins(4, 4, 4, 4)

        self.coord_label = QLabel("当前坐标: (-, -)")
        self.energy_label = QLabel("当前能量 E: -")
        self.grad_norm_label = QLabel("梯度模长 |∇E|: -")
        self.eigenvalue_label = QLabel("Hessian本征值: λ₁=-, λ₂=-")
        self.iter_label = QLabel("迭代步数: 0")

        physics_layout.addWidget(self.coord_label)
        physics_layout.addWidget(self.energy_label)
        physics_layout.addWidget(self.grad_norm_label)
        physics_layout.addWidget(self.eigenvalue_label)
        physics_layout.addWidget(self.iter_label)

        # Dimer额外信息
        self.dimer_info_widget = QWidget()
        dimer_layout = QVBoxLayout(self.dimer_info_widget)
        dimer_layout.setContentsMargins(0, 0, 0, 0)
        dimer_layout.setSpacing(1)
        self.dimer_dir_label = QLabel("双子方向 N̂: -")
        self.dimer_curv_label = QLabel("曲率 C: -")
        self.dimer_fpar_label = QLabel("平行力 |F∥|: -")
        self.dimer_fperp_label = QLabel("垂直力 |F⊥|: -")
        dimer_layout.addWidget(self.dimer_dir_label)
        dimer_layout.addWidget(self.dimer_curv_label)
        dimer_layout.addWidget(self.dimer_fpar_label)
        dimer_layout.addWidget(self.dimer_fperp_label)
        self.dimer_info_widget.setVisible(False)
        physics_layout.addWidget(self.dimer_info_widget)

        # NEB额外信息
        self.neb_info_widget = QWidget()
        neb_info_layout = QVBoxLayout(self.neb_info_widget)
        neb_info_layout.setContentsMargins(0, 0, 0, 0)
        neb_info_layout.setSpacing(1)
        self.neb_max_e_label = QLabel("路径最高能量点: -")
        self.neb_barrier_label = QLabel("粗略能垒: -")
        self.neb_ci_pos_label = QLabel("CI-NEB过渡态位置: -")
        neb_info_layout.addWidget(self.neb_max_e_label)
        neb_info_layout.addWidget(self.neb_barrier_label)
        neb_info_layout.addWidget(self.neb_ci_pos_label)
        self.neb_info_widget.setVisible(False)
        physics_layout.addWidget(self.neb_info_widget)

        physics_layout.addStretch()
        physics_group.setLayout(physics_layout)
        main_layout.addWidget(physics_group)

        # ---- 势能面公式面板 ----
        pes_formula_group = QGroupBox("势能面公式")
        pes_formula_group.setProperty("flat", True)
        pes_formula_group.setObjectName("infoPESFormula")
        pes_formula_layout = QVBoxLayout()
        pes_formula_layout.setContentsMargins(4, 4, 4, 4)
        self.pes_formula_browser = QTextBrowser()
        self.pes_formula_browser.setOpenExternalLinks(False)
        pes_formula_layout.addWidget(self.pes_formula_browser)
        pes_formula_group.setLayout(pes_formula_layout)
        main_layout.addWidget(pes_formula_group, stretch=1)

        # ---- 算法原理面板 ----
        principle_group = QGroupBox("算法原理")
        principle_group.setProperty("flat", True)
        principle_group.setObjectName("infoPrinciple")
        principle_layout = QVBoxLayout()
        principle_layout.setContentsMargins(4, 4, 4, 4)
        self.principle_browser = QTextBrowser()
        self.principle_browser.setOpenExternalLinks(False)
        principle_layout.addWidget(self.principle_browser)
        principle_group.setLayout(principle_layout)
        main_layout.addWidget(principle_group, stretch=2)

        # ---- 关键术语/误区 ----
        terms_mistakes_group = QGroupBox("关键术语/误区")
        terms_mistakes_group.setProperty("flat", True)
        terms_mistakes_group.setObjectName("infoTerms")
        terms_mistakes_layout = QVBoxLayout()
        terms_mistakes_layout.setContentsMargins(4, 4, 4, 4)
        self.terms_browser = QTextBrowser()
        self.terms_browser.setOpenExternalLinks(False)
        self.terms_browser.setHtml(self._get_terms_html())
        terms_mistakes_layout.addWidget(self.terms_browser)
        self.mistakes_browser = QTextBrowser()
        self.mistakes_browser.setOpenExternalLinks(False)
        self.mistakes_browser.setHtml(self._get_mistakes_html())
        terms_mistakes_layout.addWidget(self.mistakes_browser)
        terms_mistakes_group.setLayout(terms_mistakes_layout)
        main_layout.addWidget(terms_mistakes_group, stretch=2)

        # ---- 结果统计区 ----
        result_group = QGroupBox("结果统计")
        result_group.setProperty("flat", True)
        result_group.setObjectName("infoResult")
        result_layout = QVBoxLayout()
        result_layout.setSpacing(2)
        result_layout.setContentsMargins(4, 4, 4, 4)
        self.result_converge_label = QLabel("收敛步数: -")
        self.result_energy_label = QLabel("最终能量: -")
        self.result_type_label = QLabel("收敛类型: -")
        result_layout.addWidget(self.result_converge_label)
        result_layout.addWidget(self.result_energy_label)
        result_layout.addWidget(self.result_type_label)

        result_layout.addStretch()
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        # 收集所有需要随高度缩放字体的QLabel
        self._scaling_labels = [
            self.coord_label, self.energy_label, self.grad_norm_label,
            self.eigenvalue_label, self.iter_label,
            self.dimer_dir_label, self.dimer_curv_label,
            self.dimer_fpar_label, self.dimer_fperp_label,
            self.neb_max_e_label, self.neb_barrier_label, self.neb_ci_pos_label,
            self.result_converge_label, self.result_energy_label, self.result_type_label,
        ]
        # 关键术语/误区浏览器（字号稍大）
        self._terms_browsers = [self.terms_browser, self.mistakes_browser]
        # 其他文本浏览器（普通缩放）
        self._other_browsers = [self.pes_formula_browser, self.principle_browser]
        # 初始应用一次字号
        self._update_font_sizes()

    def resizeEvent(self, event):
        """随面板高度变化同比例缩放字体"""
        super().resizeEvent(event)
        self._update_font_sizes()

    def _update_font_sizes(self):
        """根据当前面板高度按比例缩放所有字体"""
        h = max(self.height(), self.minimumHeight())
        scale = h / self.BASE_HEIGHT
        # 普通标签字号
        font_size = max(7, int(self.BASE_FONT_SIZE * scale))
        # 关键术语/误区字号（比普通标签大）
        terms_font_size = max(8, int(self.BASE_TERMS_FONT_SIZE * scale))
        # 更新普通QLabel
        font = QFont()
        font.setPointSize(font_size)
        for label in self._scaling_labels:
            label.setFont(font)
        # 更新关键术语/误区浏览器（字号稍大）
        terms_font = QFont()
        terms_font.setPointSize(terms_font_size)
        for browser in self._terms_browsers:
            browser.setFont(terms_font)
        # 更新其他文本浏览器
        for browser in self._other_browsers:
            browser.setFont(font)
        # 重新设置关键术语/误区HTML以应用新的字号样式
        self.terms_browser.setHtml(self._get_terms_html(terms_font_size))
        self.mistakes_browser.setHtml(self._get_mistakes_html(terms_font_size))

    def _get_terms_html(self, font_size=None):
        """关键术语解释HTML"""
        if font_size is None:
            font_size = self.BASE_TERMS_FONT_SIZE
        return f"""
        <style>
            dt {{ font-weight: bold; color: #0066cc; margin-top: 2px; font-size: {font_size + 1}px; }}
            dd {{ margin-left: 10px; font-size: {font_size}px; }}
        </style>
        <dl>
        <dt>势能面 (PES)</dt>
        <dd>描述体系能量随坐标变化的超曲面。</dd>
        <dt>极小值</dt>
        <dd>Hessian正定（所有本征值>0）的驻点。</dd>
        <dt>鞍点</dt>
        <dd>一个负本征值的驻点，对应过渡态。</dd>
        <dt>Hessian矩阵</dt>
        <dd>能量二阶导数矩阵，本征值符号决定驻点类型。</dd>
        </dl>
        """

    def _get_mistakes_html(self, font_size=None):
        """常见误区HTML"""
        if font_size is None:
            font_size = self.BASE_TERMS_FONT_SIZE
        return f"""
        <ul style="font-size:{font_size}px; color:#cc3300; margin:2px;">
        <li>初始点离鞍点太远，牛顿法会收敛到极小值</li>
        <li>Dimer初始方向错误会找不到过渡态</li>
        <li>NEB镜像点太少会导致路径不光滑</li>
        <li>忽略Hessian本征值符号会错误判断驻点类型</li>
        </ul>
        """

    def update_physics(self, state, algo_type):
        """更新实时物理量显示"""
        if state is None:
            return

        # 通用信息
        if 'position' in state:
            x, y = state['position']
        elif 'center' in state:
            x, y = state['center']
        else:
            x, y = 0, 0

        self.coord_label.setText(f"当前坐标: ({x:.4f}, {y:.4f})")
        self.energy_label.setText(f"当前能量 E: {state.get('energy', 0):.6f}")
        self.grad_norm_label.setText(
            f"梯度模长 |∇E|: {state.get('gradient_norm', 0):.6f}"
        )

        if 'hessian_eigenvalues' in state:
            ev = state['hessian_eigenvalues']
            self.eigenvalue_label.setText(
                f"Hessian本征值: λ₁={ev[0]:.4f}, λ₂={ev[1]:.4f}"
            )

        self.iter_label.setText(f"迭代步数: {state.get('iteration', 0)}")

        # Dimer/CBD额外信息
        is_dimer = algo_type in ("Dimer方法", "CBD方法")
        self.dimer_info_widget.setVisible(is_dimer)
        if is_dimer:
            d = state.get('direction', np.array([0, 0]))
            self.dimer_dir_label.setText(f"双子方向 N̂: ({d[0]:.4f}, {d[1]:.4f})")
            self.dimer_curv_label.setText(f"曲率 C: {state.get('curvature', 0):.6f}")
            self.dimer_fpar_label.setText(f"平行力 |F∥|: {state.get('f_parallel', 0):.6f}")
            self.dimer_fperp_label.setText(f"垂直力 |F⊥|: {state.get('f_perp', 0):.6f}")

        # NEB额外信息
        is_neb = algo_type in ("NEB方法", "CI-NEB方法")
        self.neb_info_widget.setVisible(is_neb)
        if is_neb and 'energies' in state:
            energies = state['energies']
            max_idx = np.argmax(energies)
            self.neb_max_e_label.setText(
                f"路径最高能量点: E={energies[max_idx]:.4f}"
            )
            barrier = energies[max_idx] - energies[0]
            self.neb_barrier_label.setText(f"粗略能垒: {barrier:.4f}")
            ci_idx = state.get('climbing_image_idx')
            if ci_idx is not None:
                images = state.get('images', [])
                if ci_idx < len(images):
                    ci_pos = images[ci_idx]
                    self.neb_ci_pos_label.setText(
                        f"CI-NEB过渡态位置: ({ci_pos[0]:.4f}, {ci_pos[1]:.4f})"
                    )
            else:
                self.neb_ci_pos_label.setText("CI-NEB过渡态位置: -")

    def update_principle(self, algo_type):
        """更新算法原理说明"""
        html_map = {
            "牛顿法(极小值)": self._newton_min_html(),
            "牛顿法(鞍点)": self._newton_saddle_html(),
            "Dimer方法": self._dimer_html(),
            "NEB方法": self._neb_html(),
            "CI-NEB方法": self._cineb_html(),
            "最速下降法": self._sd_html(),
            "盆地跳跃法": self._bh_html(),
            "Metropolis MC": self._mc_html(),
            "SSW方法": self._sSW_html(),
            "元动力学": self._meta_html(),
            "极小点跳跃": self._mh_html(),
            "遗传算法": self._ga_html(),
            "粒子群优化": self._pso_html(),
            "人工蜂群": self._abc_html(),
            "伞形采样": self._us_html(),
            "自适应偏置力": self._abf_html(),
            "CBD方法": self._cbd_html(),
            "DESW方法": self._desw_html(),
        }
        self.principle_browser.setHtml(html_map.get(algo_type, ""))

    def update_pes_formula(self, pes):
        """更新势能面公式显示"""
        if pes is None:
            self.pes_formula_browser.setHtml("")
            return
        formula_map = {
            "Müller-Brown": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = Σᵢ Aᵢ·exp(aᵢ(x-x₀ᵢ)² + bᵢ(x-x₀ᵢ)(y-y₀ᵢ) + cᵢ(y-y₀ᵢ)²)</p>
            <p>经典测试函数，含3个极小值和2个鞍点</p>
            """,
            "双阱": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = (x²-1)² + y²</p>
            <p>双阱势能面，两个极小值在(±1,0)</p>
            """,
            "三极小值": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = x⁴/4 - x²/2 + y²/2</p>
            <p>三极小值势能面</p>
            """,
            "Rosenbrock": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = (1-x)² + 100(y-x²)²</p>
            <p>香蕉函数，狭长弯曲谷底，全局极小值在(1,1)</p>
            """,
            "Himmelblau": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = (x²+y-11)² + (x+y²-7)²</p>
            <p>四个等价极小值</p>
            """,
            "Ackley": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = -20·exp(-0.2·√(0.5(x²+y²))) - exp(0.5(cos2πx+cos2πy)) + e + 20</p>
            <p>大量局部极小值，全局极小值在原点</p>
            """,
            "Rastrigin": """
            <p style="text-align:center; font-weight:bold; color:#00d4ff;">
            E = 20 + x²-10cos(2πx) + y²-10cos(2πy)</p>
            <p>规则分布的大量局部极小值</p>
            """,
        }
        name = pes.name if hasattr(pes, 'name') else ""
        html = formula_map.get(name, f"<p>自定义势能面: {getattr(pes, '_expression', name)}</p>")
        self.pes_formula_browser.setHtml(html)

    @staticmethod
    def _wrap_html(body):
        """用暗色主题样式包裹HTML内容"""
        return f"""<div style="color:#c8cce0; font-size:9pt;">
        <style>h4{{color:#00d4ff; margin:2px 0;}} b{{color:#00d4ff;}}</style>
        {body}</div>"""

    def _newton_min_html(self):
        return self._wrap_html("""
        <h4>牛顿法（极小值搜索）</h4>
        <p>迭代公式：</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        q<sub>t+1</sub> = q<sub>t</sub> + H<sup>-1</sup>·F</p>
        <p>其中 F = -∇E 为力，H 为 Hessian 矩阵。</p>
        <p><b>关键：</b>正定 Hessian → 收敛到极小值。</p>
        """)

    def _newton_saddle_html(self):
        return self._wrap_html("""
        <h4>牛顿法（鞍点搜索）</h4>
        <p>反转 Hessian 最小本征值对应的分量：</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        H' = P·Λ'·P<sup>T</sup>，最小本征值取反</p>
        <p><b>关键：</b>反转后有负本征值 → 收敛到鞍点。</p>
        """)

    def _dimer_html(self):
        return self._wrap_html("""
        <h4>Dimer方法</h4>
        <p>通过双子数值求解 Hessian 最小本征值。</p>
        <p><b>旋转：</b>寻找最低曲率方向</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        C = (F₁-F₂)·N̂ / (2ΔR)</p>
        <p><b>平动：</b>F† = -F∥ + F⊥</p>
        """)

    def _neb_html(self):
        return self._wrap_html("""
        <h4>NEB方法</h4>
        <p>在初态和终态间生成镜像点，弹簧力+真实力优化路径。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        F<sup>NEB</sup> = F⊥ + F<sup>S</sup>∥</p>
        """)

    def _cineb_html(self):
        return self._wrap_html("""
        <h4>CI-NEB方法</h4>
        <p>对最高能量镜像点(CI)：移除弹簧力，反转切向力。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        F<sup>CI</sup> = F⊥ - (F·τ̂)·τ̂</p>
        """)

    def _sd_html(self):
        return self._wrap_html("""
        <h4>最速下降法</h4>
        <p>沿负梯度方向逐步搜索极小值。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        q<sub>t+1</sub> = q<sub>t</sub> - α·∇E(q<sub>t</sub>)</p>
        <p>其中 α 为步长，∇E 为梯度。</p>
        <p><b>关键：</b>每步沿能量下降最快的方向移动，简单但收敛较慢。</p>
        """)

    def _bh_html(self):
        return self._wrap_html("""
        <h4>盆地跳跃法</h4>
        <p>随机扰动 + 局部优化 + Metropolis接受准则。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        接受概率 P = min(1, exp(-ΔE/kT))</p>
        <p>对当前构型施加随机扰动后进行局部极小化，</p>
        <p>根据Metropolis准则决定是否接受新构型。</p>
        <p><b>关键：</b>能够越过能量势垒，高效搜索全局极小。</p>
        """)

    def _mc_html(self):
        return self._wrap_html("""
        <h4>Metropolis MC方法</h4>
        <p>基于Boltzmann分布的随机采样方法。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        P<sub>accept</sub> = min(1, exp(-(E<sub>new</sub>-E<sub>old</sub>)/kT))</p>
        <p>随机扰动当前构型，若能量降低则接受，</p>
        <p>若能量升高则以Boltzmann概率接受。</p>
        <p><b>关键：</b>温度kT控制采样广度，高温利于探索，低温利于收敛。</p>
        """)

    def _sSW_html(self):
        return self._wrap_html("""
        <h4>SSW方法（随机势能面行走）</h4>
        <p>随机位移 + 高斯偏置势"爬山" + 局部优化 + Metropolis接受。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        V<sub>bias</sub>(R) = Σ A<sub>i</sub> · exp(-|R - R<sub>i</sub>|² / (2σ²))</p>
        <p>沿随机方向添加高斯偏置势驱使体系翻越势垒，</p>
        <p>局部优化落入新极小点后以Metropolis准则判断接受。</p>
        <p><b>关键：</b>高斯偏置势帮助跨越势垒，MC温度控制接受概率。</p>
        """)

    def _meta_html(self):
        return self._wrap_html("""
        <h4>元动力学（MetaDynamics）</h4>
        <p>在反应坐标上持续添加高斯排斥势，驱使体系逃离局部极小。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        V<sub>bias</sub>(ξ, t) = Σ w · exp(-(ξ - ξ(t'))² / (2δξ²))</p>
        <p>每步在当前位置添加高斯势，逐渐填满当前能量井，</p>
        <p>迫使体系向未探索区域移动。</p>
        <p><b>关键：</b>高斯高度控制填充速度，高斯宽度控制精度。</p>
        """)

    def _mh_html(self):
        return self._wrap_html("""
        <h4>极小点跳跃（Minima Hopping）</h4>
        <p>短时分子动力学逃离 + 局部优化 + 自适应接受准则。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        初始动能 → MD逃逸 → 局部优化 → 接受/拒绝</p>
        <p>通过分子动力学赋予体系动能翻越势垒，</p>
        <p>动态调整动能和接受阈值平衡探索与收敛。</p>
        <p><b>关键：</b>初始动能决定逃逸能力，自适应机制保证效率。</p>
        """)

    def _ga_html(self):
        return self._wrap_html("""
        <h4>遗传算法（Genetic Algorithm）</h4>
        <p>模拟自然选择：选择 → 交叉 → 变异 → 局部优化。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        选择: 适应度高的个体更可能被选中</p>
        <p>BLX-α交叉产生子代，高斯变异增加多样性，</p>
        <p>每代个体经局部优化后替换最差个体。</p>
        <p><b>关键：</b>种群大小和变异率影响搜索广度与收敛速度。</p>
        """)

    def _pso_html(self):
        return self._wrap_html("""
        <h4>粒子群优化（PSO）</h4>
        <p>群体智能优化：每个粒子根据自身和全局最优更新位置。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        v<sub>i</sub> = w·v<sub>i</sub> + c₁·r₁·(pbest - x<sub>i</sub>) + c₂·r₂·(gbest - x<sub>i</sub>)</p>
        <p>惯性权重w平衡全局与局部搜索，</p>
        <p>认知系数c₁和社会系数c₂控制个体与群体经验的影响。</p>
        <p><b>关键：</b>惯性权重、认知/社会系数共同决定搜索行为。</p>
        """)

    def _abc_html(self):
        return self._wrap_html("""
        <h4>人工蜂群算法（ABC）</h4>
        <p>模拟蜜蜂觅食：雇佣蜂 → 观察蜂 → 侦察蜂。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        雇佣蜂: x<sub>new</sub> = x<sub>i</sub> + φ·(x<sub>i</sub> - x<sub>k</sub>)</p>
        <p>观察蜂按适应度轮盘赌选择食物源，</p>
        <p>侦察蜂在食物源耗尽后随机探索新区域。</p>
        <p><b>关键：</b>放弃阈值控制探索与开发的平衡。</p>
        """)

    def _us_html(self):
        return self._wrap_html("""
        <h4>伞形采样（Umbrella Sampling）</h4>
        <p>在反应坐标上设置谐振偏置窗口，限制采样区间。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        V<sub>bias</sub>(ξ) = ½ k (ξ - ξ₀)²</p>
        <p>各窗口独立采样后通过WHAM合并，</p>
        <p>得到完整自由能曲线。偏置势将体系约束在窗口中心附近。</p>
        <p><b>关键：</b>弹簧常数k和窗口数决定采样精度与覆盖度。</p>
        """)

    def _abf_html(self):
        return self._wrap_html("""
        <h4>自适应偏置力法（ABF）</h4>
        <p>沿反应坐标累积平均力，施加反向偏置力抵消自由能梯度。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        V<sub>bias</sub>(ξ) = -∫ F<sub>avg</sub>(ξ') dξ'</p>
        <p>随采样进行，偏置势逐渐平坦化自由能面，</p>
        <p>使体系沿反应坐标自由扩散，积分得到自由能剖面。</p>
        <p><b>关键：</b>分箱数和采样步数影响力的估计精度。</p>
        """)

    def _cbd_html(self):
        return self._wrap_html("""
        <h4>约束布罗伊登双子法（CBD）</h4>
        <p>双子数值计算最低曲率方向，Broyden更新加速收敛。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        λ = (F₁-F₂)·N̂ / ΔR，F<sub>eff</sub> = F⊥ - λ·N̂</p>
        <p>旋转双子找虚频方向，沿该方向爬升到鞍点，</p>
        <p>Broyden更新避免每次精确计算Hessian。</p>
        <p><b>关键：</b>双子间距ΔR影响曲率计算精度。</p>
        """)

    def _desw_html(self):
        return self._wrap_html("""
        <h4>双端行走法（DESW）</h4>
        <p>从初态和终态分别出发，添加高斯偏置势行走至相遇。</p>
        <p style="text-align:center; font-weight:bold; color:#00d4ff;">
        V<sub>bias</sub>(R) = Σ A<sub>i</sub>·exp(-|R-R<sub>i</sub>|²/(2σ²))</p>
        <p>两端交替行走，高斯偏置帮助翻越势垒，</p>
        <p>轨迹相遇后连接路径，取最高点为过渡态猜测。</p>
        <p><b>关键：</b>双端约束保证路径物理合理性。</p>
        """)

    def update_result(self, trajectory, algo_type, pes):
        """更新结果统计"""
        if not trajectory:
            self.result_converge_label.setText("收敛步数: -")
            self.result_energy_label.setText("最终能量: -")
            self.result_type_label.setText("收敛类型: -")
            return

        last = trajectory[-1]
        n_steps = len(trajectory)
        final_energy = last.get('energy', 0)

        # 判断收敛类型
        if 'hessian_eigenvalues' in last:
            ev = last['hessian_eigenvalues']
            if np.all(ev > 0):
                conv_type = "极小值"
            elif np.sum(ev < 0) == 1:
                conv_type = "过渡态(鞍点)"
            else:
                conv_type = "发散"
        elif 'curvature' in last:
            if last.get('curvature', 0) < 0:
                conv_type = "过渡态(鞍点)"
            else:
                conv_type = "未找到鞍点"
        else:
            conv_type = "路径优化"

        self.result_converge_label.setText(f"收敛步数: {n_steps}")
        self.result_energy_label.setText(f"最终能量: {final_energy:.6f}")
        self.result_type_label.setText(f"收敛类型: {conv_type}")

    def show_compare_table(self, results):
        """显示方法对比表格（已移至主对比面板，此方法保留为空以兼容）"""
        pass

    def reset(self):
        """重置信息面板"""
        self.coord_label.setText("当前坐标: (-, -)")
        self.energy_label.setText("当前能量 E: -")
        self.grad_norm_label.setText("梯度模长 |∇E|: -")
        self.eigenvalue_label.setText("Hessian本征值: λ₁=-, λ₂=-")
        self.iter_label.setText("迭代步数: 0")
        self.dimer_info_widget.setVisible(False)
        self.neb_info_widget.setVisible(False)
        self.result_converge_label.setText("收敛步数: -")
        self.result_energy_label.setText("最终能量: -")
        self.result_type_label.setText("收敛类型: -")


# ======================== 主窗口 ========================

class MainWindow(QMainWindow):
    """主窗口：整合布局和所有交互逻辑"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("势能面搜索算法交互式可视化教学工具 — TFAA@2026")
        self.resize(1400, 900)

        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pesvisualizerlogo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # 状态变量
        self.pes = None
        self.algorithm = None
        self.compare_algorithm = None
        self.start_pos = None
        self.end_pos = None
        self.click_mode = 'start'  # 'start' 或 'end'
        self._point_setting_mode = False  # 是否处于点设置模式
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._timer_step)
        self.trajectory = []
        self.compare_trajectory = []
        self._cmp_converge_notified = False  # 对比算法收敛提示标志，避免重复弹框
        self._main_converge_notified = False  # 主算法收敛提示标志（对比模式下）

        self._init_ui()
        self._connect_signals()
        self._load_pes()

    def _init_ui(self):
        """初始化UI"""
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setSpacing(2)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        # 上方区域：左侧控制面板 | 中间画布（水平分割）
        upper_splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        self.control_panel = ControlPanel()
        upper_splitter.addWidget(self.control_panel)

        # 中间可视化区域
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(0)

        self.canvas = PESCanvas()
        self.canvas.setFocusPolicy(Qt.StrongFocus)  # 允许接收键盘事件
        self.canvas._redraw_callback = self._update_view  # 网格扩展时重绘
        # 第二个画布（用于方法对比）
        self.canvas2 = PESCanvas()
        self.canvas2.setFocusPolicy(Qt.StrongFocus)
        self.canvas2._redraw_callback = self._update_view
        self.canvas2.setVisible(False)  # 默认隐藏

        # 水平布局放置两个画布
        canvas_h_layout = QHBoxLayout()
        canvas_h_layout.setContentsMargins(0, 0, 0, 0)
        canvas_h_layout.setSpacing(2)
        canvas_h_layout.addWidget(self.canvas, 1)  # stretch=1 确保等大
        canvas_h_layout.addWidget(self.canvas2, 1)  # stretch=1 确保等大
        viz_layout.addLayout(canvas_h_layout)
        upper_splitter.addWidget(viz_widget)

        # 设置分割比例 - 允许拖动调整
        upper_splitter.setSizes([300, 1100])
        upper_splitter.setStretchFactor(0, 0)  # 控制面板不自动拉伸
        upper_splitter.setStretchFactor(1, 1)  # 画布自动拉伸
        upper_splitter.setChildrenCollapsible(False)  # 不允许完全折叠

        # 下方信息面板
        self.info_panel = InfoPanel()

        # 方法对比面板（位于画布和信息面板之间）
        self.compare_panel = QWidget()
        self.compare_panel.setVisible(False)
        self.compare_panel.setMinimumHeight(140)  # 确保内容可见
        compare_panel_layout = QVBoxLayout(self.compare_panel)
        compare_panel_layout.setContentsMargins(4, 2, 4, 2)
        compare_panel_layout.setSpacing(2)
        compare_title = QLabel("方法对比结果")
        compare_title.setStyleSheet("font-weight: bold; color: #00d4ff;")
        compare_panel_layout.addWidget(compare_title)
        # 复用InfoPanel的compare_table样式，但这里独立创建
        self.compare_table_main = QTableWidget()
        self.compare_table_main.setMinimumHeight(100)
        self.compare_table_main.setVisible(False)
        compare_panel_layout.addWidget(self.compare_table_main)

        # 垂直分割：上方 | 对比面板 | 下方 - 允许拖动调整
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(upper_splitter)
        main_splitter.addWidget(self.compare_panel)
        main_splitter.addWidget(self.info_panel)
        main_splitter.setSizes([500, 160, 200])
        main_splitter.setStretchFactor(0, 1)  # 上方自动拉伸
        main_splitter.setStretchFactor(1, 0)  # 对比面板不自动拉伸
        main_splitter.setStretchFactor(2, 0)  # 下方不自动拉伸
        main_splitter.setChildrenCollapsible(False)  # 不允许完全折叠

        outer_layout.addWidget(main_splitter)

    def _connect_signals(self):
        """连接信号和槽"""
        cp = self.control_panel

        # 势能面切换
        cp.pes_combo.currentIndexChanged.connect(self._load_pes)
        cp.custom_expr_confirm_btn.clicked.connect(self._load_custom_pes)
        cp.apply_range_btn.clicked.connect(self._apply_range)

        # 范围输入框回车确认
        for spin in [cp.x_min_spin, cp.x_max_spin, cp.y_min_spin, cp.y_max_spin]:
            spin.editingFinished.connect(self._apply_range)
        # 自定义表达式输入框回车确认
        cp.custom_expr.returnPressed.connect(self._load_custom_pes)

        # 渲染模式切换
        cp.contour_check.stateChanged.connect(self._update_view)
        cp.surface3d_check.stateChanged.connect(self._update_view)

        # 算法切换
        cp.algo_combo.currentIndexChanged.connect(self._on_algo_changed)

        # 算法参数确认按钮
        cp.newton_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.dimer_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.neb_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.sd_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.bh_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.mc_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.sSW_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.meta_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.mh_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.ga_confirm_btn.clicked.connect(self._confirm_algo_params)
        cp.pso_confirm_btn.clicked.connect(self._confirm_algo_params)

        # 参数输入框回车确认
        for spin in [cp.newton_threshold, cp.newton_max_iter,
                     cp.dimer_delta_r, cp.dimer_threshold, cp.dimer_max_iter, cp.dimer_trans_step,
                     cp.neb_n_images, cp.neb_spring_k, cp.neb_threshold, cp.neb_max_iter,
                     cp.sd_step_size, cp.sd_threshold, cp.sd_max_iter,
                     cp.bh_step_size, cp.bh_temperature, cp.bh_max_iter,
                     cp.mc_step_size, cp.mc_temperature, cp.mc_max_iter,
                     cp.sSW_step_size, cp.sSW_gaussian_height, cp.sSW_gaussian_width, cp.sSW_temperature, cp.sSW_max_iter,
                     cp.meta_gaussian_height, cp.meta_gaussian_width, cp.meta_max_iter,
                     cp.mh_kinetic, cp.mh_max_iter,
                     cp.ga_pop_size, cp.ga_mutation_rate, cp.ga_max_iter, cp.ga_search_range,
                     cp.pso_n_particles, cp.pso_w, cp.pso_c1, cp.pso_c2, cp.pso_max_iter,
                     cp.abc_n_bees, cp.abc_limit, cp.abc_max_iter, cp.abc_search_range,
                     cp.us_n_windows, cp.us_spring_k, cp.us_mc_step_size, cp.us_mc_temperature, cp.us_steps_per_window, cp.us_max_iter,
                     cp.abf_n_bins, cp.abf_mc_step_size, cp.abf_mc_temperature, cp.abf_max_iter,
                     cp.cbd_delta_r, cp.cbd_threshold, cp.cbd_trans_step, cp.cbd_max_iter,
                     cp.desw_step_size, cp.desw_gaussian_height, cp.desw_gaussian_width, cp.desw_temperature, cp.desw_max_iter]:
            if hasattr(spin, 'editingFinished'):
                spin.editingFinished.connect(self._confirm_algo_params)

        # 初始点设置
        cp.set_point_btn.clicked.connect(self._enter_point_setting)
        cp.finish_point_btn.clicked.connect(self._exit_point_setting)
        # 输入框回车键设置初始点
        cp.init_x1.returnPressed.connect(self._set_init_from_input)
        cp.init_y1.returnPressed.connect(self._set_init_from_input)
        cp.init_x2.returnPressed.connect(self._set_init_from_input)
        cp.init_y2.returnPressed.connect(self._set_init_from_input)

        # 播放控制
        cp.play_btn.clicked.connect(self._toggle_play)
        cp.step_btn.clicked.connect(self._single_step)
        cp.reset_btn.clicked.connect(self._reset)
        cp.back_btn.clicked.connect(self._back_step)
        cp.help_btn.clicked.connect(self._show_help)
        cp.compare_check.stateChanged.connect(self._on_compare_mode_changed)
        cp.compare_edit_params_btn.clicked.connect(self._toggle_compare_params_edit)
        cp.compare_algo_combo.currentTextChanged.connect(self._on_compare_algo_changed)

        # 导出
        cp.export_csv_btn.clicked.connect(self._export_csv)
        cp.export_png_btn.clicked.connect(self._export_png)
        cp.export_gif_btn.clicked.connect(self._export_gif)

        # 鼠标点击（主画布和对比画布共用同一回调）
        self.canvas.set_click_callback(self._on_canvas_click)
        self.canvas2.set_click_callback(self._on_canvas_click)

    def _load_pes(self):
        """加载势能面"""
        pes_name = self.control_panel.pes_combo.currentText()
        try:
            if pes_name == "Müller-Brown":
                self.pes = MuellerBrownPES()
            elif pes_name == "双阱":
                self.pes = DoubleWellPES()
            elif pes_name == "三极小值":
                self.pes = ThreeMinimaPES()
            elif pes_name == "Rosenbrock":
                self.pes = RosenbrockPES()
            elif pes_name == "Himmelblau":
                self.pes = HimmelblauPES()
            elif pes_name == "Ackley":
                self.pes = AckleyPES()
            elif pes_name == "Rastrigin":
                self.pes = RastriginPES()
            elif pes_name == "自定义":
                self._load_custom_pes()
                return
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载势能面失败: {e}")
            return

        # 更新范围spinbox为PES的默认bounds
        self._update_range_spins()
        # 清除自定义范围，使用新PES的默认范围
        self.canvas._custom_bounds = None
        # 重置布局以适应新PES
        self.canvas._current_layout = None  # 强制重建布局
        self.canvas._user_dist = None
        self.canvas._user_xlim = None
        self.canvas._user_ylim = None
        self.canvas._cbar_ax = None  # 重建布局后colorbar axes也需要重置
        self.canvas._reset_camera_center = True  # 切换PES后重置3D相机中心到新PES中心
        self.canvas._pan_offset[:] = 0  # 重置3D平移偏移
        # 同步重置canvas2（对比画布）
        self.canvas2._custom_bounds = None
        self.canvas2._current_layout = None
        self.canvas2._user_dist = None
        self.canvas2._user_xlim = None
        self.canvas2._user_ylim = None
        self.canvas2._cbar_ax = None
        self.canvas2._reset_camera_center = True
        self.canvas2._pan_offset[:] = 0
        self.canvas2.X = None  # 强制重新计算网格
        self.canvas2.Z = None
        self._reset_state()
        self._render_pes()

    def _load_custom_pes(self, checked=False):
        """加载自定义势能面"""
        expr = self.control_panel.custom_expr.text().strip()
        if not expr:
            return
        try:
            # 先应用隐式乘法预处理（与CustomPES.__init__中相同的逻辑）
            import re
            processed_expr = expr
            processed_expr = re.sub(r'(\d)([xy])', r'\1*\2', processed_expr)      # 2y → 2*y
            processed_expr = re.sub(r'(\d)\(', r'\1*(', processed_expr)             # 2( → 2*(
            processed_expr = re.sub(r'\)(\d)', r')*\1', processed_expr)             # )2 → )*2
            processed_expr = re.sub(r'\)\(', r')*(', processed_expr)                # )( → )*(
            processed_expr = re.sub(r'\)([xy])', r')*\1', processed_expr)           # )x → )*x
            processed_expr = re.sub(r'([xy])\(', r'\1*(', processed_expr)           # x( → x*(
            processed_expr = re.sub(r'([xy])(\d)', r'\1*\2', processed_expr)        # x2 → x*2 (罕见但可能)

            # 测试表达式是否有效（使用CustomPES的数学命名空间）
            from pes import CustomPES
            test_ns = {'x': 0.5, 'y': 0.5}
            test_ns.update(CustomPES._MATH_NAMESPACE)
            test_val = eval(processed_expr, {"__builtins__": {}}, test_ns)
            float(test_val)
            self.pes = CustomPES(expression=expr)
            self.canvas._custom_bounds = None
            self.canvas._current_layout = None  # 强制重建布局
            self.canvas._user_dist = None
            self.canvas._user_xlim = None
            self.canvas._user_ylim = None
            self.canvas._cbar_ax = None
            self.canvas._reset_camera_center = True  # 切换PES后重置3D相机中心
            self.canvas._pan_offset[:] = 0  # 重置3D平移偏移
            # 同步重置canvas2（对比画布）
            self.canvas2._custom_bounds = None
            self.canvas2._current_layout = None
            self.canvas2._user_dist = None
            self.canvas2._user_xlim = None
            self.canvas2._user_ylim = None
            self.canvas2._cbar_ax = None
            self.canvas2._reset_camera_center = True
            self.canvas2._pan_offset[:] = 0
            self.canvas2.X = None
            self.canvas2.Z = None
            self._update_range_spins()
            self._reset_state()
            self._render_pes()
        except Exception as e:
            QMessageBox.warning(self, "表达式错误", f"无法解析表达式:\n{e}")

    def _update_range_spins(self):
        """根据当前PES的bounds智能更新范围spinbox"""
        if self.pes is None:
            return
        xmin, xmax, ymin, ymax = self.pes.bounds
        cp = self.control_panel
        # 阻止信号触发，避免递归
        cp.x_min_spin.blockSignals(True)
        cp.x_max_spin.blockSignals(True)
        cp.y_min_spin.blockSignals(True)
        cp.y_max_spin.blockSignals(True)
        cp.x_min_slider.blockSignals(True)
        cp.x_max_slider.blockSignals(True)
        cp.y_min_slider.blockSignals(True)
        cp.y_max_slider.blockSignals(True)
        cp.x_min_spin.setValue(xmin)
        cp.x_max_spin.setValue(xmax)
        cp.y_min_spin.setValue(ymin)
        cp.y_max_spin.setValue(ymax)
        cp.x_min_slider.setValue(int(xmin * 100))
        cp.x_max_slider.setValue(int(xmax * 100))
        cp.y_min_slider.setValue(int(ymin * 100))
        cp.y_max_slider.setValue(int(ymax * 100))
        cp.x_min_spin.blockSignals(False)
        cp.x_max_spin.blockSignals(False)
        cp.y_min_spin.blockSignals(False)
        cp.y_max_spin.blockSignals(False)
        cp.x_min_slider.blockSignals(False)
        cp.x_max_slider.blockSignals(False)
        cp.y_min_slider.blockSignals(False)
        cp.y_max_slider.blockSignals(False)

    def _apply_range(self):
        """应用用户指定的PES显示范围"""
        if self.pes is None:
            return
        cp = self.control_panel
        xmin = cp.x_min_spin.value()
        xmax = cp.x_max_spin.value()
        ymin = cp.y_min_spin.value()
        ymax = cp.y_max_spin.value()
        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "范围错误", "最小值必须小于最大值")
            return
        # 保存自定义范围到canvas
        self.canvas._custom_bounds = (xmin, xmax, ymin, ymax)
        # 重置视角以适应新范围 - 需要重建布局
        self.canvas._user_dist = None
        self.canvas._user_xlim = None
        self.canvas._user_ylim = None
        self.canvas._current_layout = None  # 强制重建布局
        self.canvas._cbar_ax = None  # 重建布局后colorbar axes也需要重置
        self.canvas._reset_camera_center = True  # 范围变化后重置3D相机中心
        self.canvas._pan_offset[:] = 0  # 重置平移偏移
        # 同步第二个画布的范围设置
        if hasattr(self, 'canvas2'):
            self.canvas2._custom_bounds = (xmin, xmax, ymin, ymax)
            self.canvas2._user_dist = None
            self.canvas2._user_xlim = None
            self.canvas2._user_ylim = None
            self.canvas2._current_layout = None
            self.canvas2._cbar_ax = None
            self.canvas2._reset_camera_center = True
            self.canvas2._pan_offset[:] = 0
        # 不调用_reset_state()，保留当前轨迹和算法状态
        self.canvas.compute_grid(self.pes)
        if hasattr(self, 'canvas2'):
            self.canvas2.compute_grid(self.pes)
        self._update_view()

    def _render_pes(self):
        """渲染势能面"""
        if self.pes is None:
            return
        self.canvas.compute_grid(self.pes)
        # 同步第二个画布的网格（用于方法对比）
        if hasattr(self, 'canvas2'):
            self.canvas2.compute_grid(self.pes)
        self.info_panel.update_pes_formula(self.pes)
        self._update_view()

    def _update_view(self):
        """更新可视化视图"""
        if self.pes is None:
            return

        # 准备轨迹数据
        trajectory = self.trajectory if self.trajectory else None
        start_pos = self.start_pos
        end_pos = self.end_pos if self._needs_two_endpoints() else None

        # Dimer/CBD信息
        dimer_info = None
        algo_type = self.control_panel.get_algo_type()
        if algo_type in ("Dimer方法", "CBD方法") and self.algorithm is not None and self.trajectory:
            last = self.trajectory[-1]
            delta_r = self.control_panel.dimer_delta_r.value() if algo_type == "Dimer方法" else self.control_panel.cbd_delta_r.value()
            dimer_info = {
                'center': last.get('center', self.start_pos),
                'direction': last.get('direction', np.array([1, 0])),
                'delta_r': delta_r,
            }

        # NEB镜像点
        neb_images = None
        energy_profile = None
        if self._is_neb_algo() and self.algorithm is not None:
            neb_images = [img.tolist() for img in self.algorithm.images]
            if self.trajectory:
                last = self.trajectory[-1]
                if 'images' in last:
                    neb_images = [img.tolist() for img in last['images']]
                # 能量曲线
                if 'path_energy_profile' in last:
                    distances, _ = self.algorithm.get_energy_profile()
                    energy_profile = (distances, last['path_energy_profile'])

        # 群体算法种群位置
        population_positions = None
        if self.trajectory:
            last = self.trajectory[-1]
            if 'population_positions' in last:
                population_positions = last['population_positions']
            elif 'particle_positions' in last:
                population_positions = last['particle_positions']

        if self.control_panel.is_3d_mode():
            self.canvas.draw_3d(
                trajectory=trajectory,
                start_pos=start_pos,
                end_pos=end_pos,
                neb_images=neb_images,
                energy_profile=energy_profile,
                population_positions=population_positions,
            )
        else:
            self.canvas.draw_contour(
                trajectory=trajectory,
                start_pos=start_pos,
                end_pos=end_pos,
                dimer_info=dimer_info,
                neb_images=neb_images,
                energy_profile=energy_profile,
                population_positions=population_positions,
            )

        # 方法对比模式：在第二个画布绘制对比算法的轨迹
        if self.control_panel.compare_check.isChecked():
            self._update_compare_view()

    def _update_compare_view(self):
        """更新方法对比的第二个画布视图"""
        if self.pes is None or not self.control_panel.compare_check.isChecked():
            return
        # 确保canvas2已计算网格
        if self.canvas2.Z is None:
            self.canvas2.compute_grid(self.pes)

        # 准备对比轨迹数据
        cmp_trajectory = self.compare_trajectory if self.compare_trajectory else None
        start_pos = self.start_pos
        end_pos = self.end_pos if self._needs_two_endpoints() else None

        # Dimer/CBD信息（对比算法）
        cmp_dimer_info = None
        cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
        if cmp_algo_type in ("Dimer方法", "CBD方法") and self.compare_algorithm is not None and self.compare_trajectory:
            last = self.compare_trajectory[-1]
            delta_r = self.control_panel.dimer_delta_r.value() if cmp_algo_type == "Dimer方法" else self.control_panel.cbd_delta_r.value()
            cmp_dimer_info = {
                'center': last.get('center', self.start_pos),
                'direction': last.get('direction', np.array([1, 0])),
                'delta_r': delta_r,
            }

        # NEB镜像点（对比算法）
        cmp_neb_images = None
        cmp_energy_profile = None
        if cmp_algo_type in ("NEB方法", "CI-NEB方法") and self.compare_algorithm is not None:
            cmp_neb_images = [img.tolist() for img in self.compare_algorithm.images]
            if self.compare_trajectory:
                last = self.compare_trajectory[-1]
                if 'images' in last:
                    cmp_neb_images = [img.tolist() for img in last['images']]
                if 'path_energy_profile' in last:
                    distances, _ = self.compare_algorithm.get_energy_profile()
                    cmp_energy_profile = (distances, last['path_energy_profile'])

        # 群体算法种群位置（对比算法）
        cmp_population = None
        if self.compare_trajectory:
            last = self.compare_trajectory[-1]
            if 'population_positions' in last:
                cmp_population = last['population_positions']
            elif 'particle_positions' in last:
                cmp_population = last['particle_positions']

        if self.control_panel.is_3d_mode():
            self.canvas2.draw_3d(
                trajectory=cmp_trajectory,
                start_pos=start_pos,
                end_pos=end_pos,
                neb_images=cmp_neb_images,
                energy_profile=cmp_energy_profile,
                population_positions=cmp_population,
            )
        else:
            self.canvas2.draw_contour(
                trajectory=cmp_trajectory,
                start_pos=start_pos,
                end_pos=end_pos,
                dimer_info=cmp_dimer_info,
                neb_images=cmp_neb_images,
                energy_profile=cmp_energy_profile,
                population_positions=cmp_population,
            )

    def _on_algo_changed(self):
        """算法切换时更新"""
        algo_type = self.control_panel.get_algo_type()
        self.info_panel.update_principle(algo_type)
        # 重置初始点
        if self._needs_two_endpoints():
            self.click_mode = 'start'
        else:
            self.click_mode = 'start'
        self._reset_state()
        self._update_view()

    def _confirm_algo_params(self):
        """确认算法参数输入（提交所有spinbox的当前文本值）"""
        cp = self.control_panel
        # 对所有QDoubleSpinBox调用interpretText，确保输入框中的文本被解析为数值
        for spin in [cp.newton_threshold, cp.newton_max_iter,
                     cp.dimer_delta_r, cp.dimer_threshold, cp.dimer_max_iter, cp.dimer_trans_step,
                     cp.neb_n_images, cp.neb_spring_k, cp.neb_threshold, cp.neb_max_iter,
                     cp.sd_step_size, cp.sd_threshold, cp.sd_max_iter,
                     cp.bh_step_size, cp.bh_temperature, cp.bh_max_iter,
                     cp.mc_step_size, cp.mc_temperature, cp.mc_max_iter,
                     cp.sSW_step_size, cp.sSW_gaussian_height, cp.sSW_gaussian_width, cp.sSW_temperature, cp.sSW_max_iter,
                     cp.meta_gaussian_height, cp.meta_gaussian_width, cp.meta_max_iter,
                     cp.mh_kinetic, cp.mh_max_iter,
                     cp.ga_pop_size, cp.ga_mutation_rate, cp.ga_max_iter, cp.ga_search_range,
                     cp.pso_n_particles, cp.pso_w, cp.pso_c1, cp.pso_c2, cp.pso_max_iter,
                     cp.abc_n_bees, cp.abc_limit, cp.abc_max_iter, cp.abc_search_range,
                     cp.us_n_windows, cp.us_spring_k, cp.us_mc_step_size, cp.us_mc_temperature, cp.us_steps_per_window, cp.us_max_iter,
                     cp.abf_n_bins, cp.abf_mc_step_size, cp.abf_mc_temperature, cp.abf_max_iter,
                     cp.cbd_delta_r, cp.cbd_threshold, cp.cbd_trans_step, cp.cbd_max_iter,
                     cp.desw_step_size, cp.desw_gaussian_height, cp.desw_gaussian_width, cp.desw_temperature, cp.desw_max_iter]:
            try:
                spin.interpretText()
            except Exception:
                pass
        # 如果已有算法实例运行中，仅重置算法实例（保留初始点）
        if self.algorithm is not None:
            self.algorithm = None
            self.trajectory = []
            self.timer.stop()
            self.is_running = False
            self.control_panel.play_btn.setText("开始")
            self.control_panel.play_btn.setObjectName("playBtn")
            self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
            self._update_view()

    def _is_neb_algo(self):
        """当前是否为NEB类算法（具有images属性的算法）"""
        return self.control_panel.get_algo_type() in ("NEB方法", "CI-NEB方法")

    def _needs_two_endpoints(self):
        """当前算法是否需要两个端点（初态+终态）"""
        return self.control_panel.get_algo_type() in ("NEB方法", "CI-NEB方法", "伞形采样", "自适应偏置力", "DESW方法")

    def _on_canvas_click(self, x, y):
        """处理画布点击事件"""
        if not self._point_setting_mode:
            return  # 非点设置模式，不处理点击
        if self._needs_two_endpoints():
            if self.click_mode == 'start':
                self.start_pos = np.array([x, y])
                self.control_panel.init_x1.setText(f"{x:.4f}")
                self.control_panel.init_y1.setText(f"{y:.4f}")
                self.click_mode = 'end'
                self.control_panel.init_hint.setText("请点击设置终态")
            elif self.click_mode == 'end':
                self.end_pos = np.array([x, y])
                self.control_panel.init_x2.setText(f"{x:.4f}")
                self.control_panel.init_y2.setText(f"{y:.4f}")
                self.click_mode = 'start'
                self.control_panel.init_hint.setText("请依次点击设置初态和终态（已设置，可重新点击）")
        else:
            self.start_pos = np.array([x, y])
            self.control_panel.init_x1.setText(f"{x:.4f}")
            self.control_panel.init_y1.setText(f"{y:.4f}")
            self.control_panel.init_hint.setText(f"初始点已设置: ({x:.4f}, {y:.4f})")

        self._update_view()

    def _set_init_from_input(self):
        """从输入框设置初始点"""
        try:
            x1 = float(self.control_panel.init_x1.text())
            y1 = float(self.control_panel.init_y1.text())
            self.start_pos = np.array([x1, y1])

            if self._needs_two_endpoints():
                x2 = float(self.control_panel.init_x2.text())
                y2 = float(self.control_panel.init_y2.text())
                self.end_pos = np.array([x2, y2])

            self._update_view()
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数值坐标")

    def _enter_point_setting(self):
        """进入点设置模式"""
        self._point_setting_mode = True
        self.control_panel.set_point_btn.setEnabled(False)
        self.control_panel.finish_point_btn.setEnabled(True)
        if self._needs_two_endpoints():
            self.control_panel.init_hint.setText("请点击设置初态，完成后点'完成'")
            self.click_mode = 'start'
        else:
            self.control_panel.init_hint.setText("请点击设置初始点，完成后点'完成'")

    def _exit_point_setting(self):
        """退出点设置模式"""
        self._point_setting_mode = False
        self.control_panel.set_point_btn.setEnabled(True)
        self.control_panel.finish_point_btn.setEnabled(False)
        if self.start_pos is not None:
            self.control_panel.init_hint.setText(
                f"初始点已设置: ({self.start_pos[0]:.4f}, {self.start_pos[1]:.4f})"
            )
        else:
            self.control_panel.init_hint.setText("点击'设置'开始选择初始点")

    def _create_algorithm(self, algo_type=None):
        """创建算法实例"""
        if algo_type is None:
            algo_type = self.control_panel.get_algo_type()

        if self.start_pos is None:
            return None

        # 提交所有spinbox的当前输入文本，确保最新值被解析
        cp = self.control_panel
        for spin in [cp.newton_threshold, cp.newton_max_iter,
                     cp.dimer_delta_r, cp.dimer_threshold, cp.dimer_max_iter, cp.dimer_trans_step,
                     cp.neb_n_images, cp.neb_spring_k, cp.neb_threshold, cp.neb_max_iter,
                     cp.sd_step_size, cp.sd_threshold, cp.sd_max_iter,
                     cp.bh_step_size, cp.bh_temperature, cp.bh_max_iter,
                     cp.mc_step_size, cp.mc_temperature, cp.mc_max_iter,
                     cp.sSW_step_size, cp.sSW_gaussian_height, cp.sSW_gaussian_width, cp.sSW_temperature, cp.sSW_max_iter,
                     cp.meta_gaussian_height, cp.meta_gaussian_width, cp.meta_max_iter,
                     cp.mh_kinetic, cp.mh_max_iter,
                     cp.ga_pop_size, cp.ga_mutation_rate, cp.ga_max_iter, cp.ga_search_range,
                     cp.pso_n_particles, cp.pso_w, cp.pso_c1, cp.pso_c2, cp.pso_max_iter,
                     cp.abc_n_bees, cp.abc_limit, cp.abc_max_iter, cp.abc_search_range,
                     cp.us_n_windows, cp.us_spring_k, cp.us_mc_step_size, cp.us_mc_temperature, cp.us_steps_per_window, cp.us_max_iter,
                     cp.abf_n_bins, cp.abf_mc_step_size, cp.abf_mc_temperature, cp.abf_max_iter,
                     cp.cbd_delta_r, cp.cbd_threshold, cp.cbd_trans_step, cp.cbd_max_iter,
                     cp.desw_step_size, cp.desw_gaussian_height, cp.desw_gaussian_width, cp.desw_temperature, cp.desw_max_iter]:
            try:
                spin.interpretText()
            except Exception:
                pass

        try:
            if algo_type in ("牛顿法(极小值)", "牛顿法(鞍点)"):
                hessian_sign = "positive" if algo_type == "牛顿法(极小值)" else "negative"
                return NewtonMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    conv_threshold=self.control_panel.newton_threshold.value(),
                    max_iter=self.control_panel.newton_max_iter.value(),
                    hessian_sign=hessian_sign,
                )
            elif algo_type == "Dimer方法":
                return DimerMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    delta_r=self.control_panel.dimer_delta_r.value(),
                    conv_threshold=self.control_panel.dimer_threshold.value(),
                    max_iter=self.control_panel.dimer_max_iter.value(),
                    trans_step=self.control_panel.dimer_trans_step.value(),
                )
            elif algo_type in ("NEB方法", "CI-NEB方法"):
                if self.end_pos is None:
                    return None
                climbing = algo_type == "CI-NEB方法"
                return NEBMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    end_pos=self.end_pos.copy(),
                    n_images=self.control_panel.neb_n_images.value(),
                    spring_k=self.control_panel.neb_spring_k.value(),
                    conv_threshold=self.control_panel.neb_threshold.value(),
                    max_iter=self.control_panel.neb_max_iter.value(),
                    climbing_image=climbing,
                )
            elif algo_type == "最速下降法":
                return SteepestDescentMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    step_size=self.control_panel.sd_step_size.value(),
                    conv_threshold=self.control_panel.sd_threshold.value(),
                    max_iter=self.control_panel.sd_max_iter.value(),
                )
            elif algo_type == "盆地跳跃法":
                return BasinHoppingMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    step_size=self.control_panel.bh_step_size.value(),
                    temperature=self.control_panel.bh_temperature.value(),
                    max_iter=self.control_panel.bh_max_iter.value(),
                )
            elif algo_type == "Metropolis MC":
                return MetropolisMCMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    step_size=self.control_panel.mc_step_size.value(),
                    temperature=self.control_panel.mc_temperature.value(),
                    max_iter=self.control_panel.mc_max_iter.value(),
                )
            elif algo_type == "SSW方法":
                return SSWMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    step_size=self.control_panel.sSW_step_size.value(),
                    gaussian_height=self.control_panel.sSW_gaussian_height.value(),
                    gaussian_width=self.control_panel.sSW_gaussian_width.value(),
                    temperature=self.control_panel.sSW_temperature.value(),
                    max_iter=self.control_panel.sSW_max_iter.value(),
                )
            elif algo_type == "元动力学":
                return MetaDynamicsMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    gaussian_height=self.control_panel.meta_gaussian_height.value(),
                    gaussian_width=self.control_panel.meta_gaussian_width.value(),
                    max_iter=self.control_panel.meta_max_iter.value(),
                )
            elif algo_type == "极小点跳跃":
                return MinimaHoppingMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    initial_kinetic=self.control_panel.mh_kinetic.value(),
                    max_iter=self.control_panel.mh_max_iter.value(),
                )
            elif algo_type == "遗传算法":
                return GeneticAlgorithmMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    population_size=self.control_panel.ga_pop_size.value(),
                    mutation_rate=self.control_panel.ga_mutation_rate.value(),
                    max_iter=self.control_panel.ga_max_iter.value(),
                    search_range=self.control_panel.ga_search_range.value(),
                )
            elif algo_type == "粒子群优化":
                return PSOMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    n_particles=self.control_panel.pso_n_particles.value(),
                    w=self.control_panel.pso_w.value(),
                    c1=self.control_panel.pso_c1.value(),
                    c2=self.control_panel.pso_c2.value(),
                    max_iter=self.control_panel.pso_max_iter.value(),
                )
            elif algo_type == "人工蜂群":
                return ABCMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    n_bees=self.control_panel.abc_n_bees.value(),
                    limit=self.control_panel.abc_limit.value(),
                    max_iter=self.control_panel.abc_max_iter.value(),
                    search_range=self.control_panel.abc_search_range.value(),
                )
            elif algo_type == "伞形采样":
                if self.end_pos is None:
                    return None
                return UmbrellaSamplingMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    end_pos=self.end_pos.copy(),
                    n_windows=self.control_panel.us_n_windows.value(),
                    spring_k=self.control_panel.us_spring_k.value(),
                    mc_step_size=self.control_panel.us_mc_step_size.value(),
                    mc_temperature=self.control_panel.us_mc_temperature.value(),
                    steps_per_window=self.control_panel.us_steps_per_window.value(),
                    max_iter=self.control_panel.us_max_iter.value(),
                )
            elif algo_type == "自适应偏置力":
                if self.end_pos is None:
                    return None
                return ABFMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    end_pos=self.end_pos.copy(),
                    n_bins=self.control_panel.abf_n_bins.value(),
                    max_iter=self.control_panel.abf_max_iter.value(),
                    mc_step_size=self.control_panel.abf_mc_step_size.value(),
                    mc_temperature=self.control_panel.abf_mc_temperature.value(),
                )
            elif algo_type == "CBD方法":
                return CBDMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    delta_r=self.control_panel.cbd_delta_r.value(),
                    conv_threshold=self.control_panel.cbd_threshold.value(),
                    max_iter=self.control_panel.cbd_max_iter.value(),
                    trans_step=self.control_panel.cbd_trans_step.value(),
                )
            elif algo_type == "DESW方法":
                if self.end_pos is None:
                    return None
                return DESWMethod(
                    pes=self.pes,
                    start_pos=self.start_pos.copy(),
                    end_pos=self.end_pos.copy(),
                    step_size=self.control_panel.desw_step_size.value(),
                    gaussian_height=self.control_panel.desw_gaussian_height.value(),
                    gaussian_width=self.control_panel.desw_gaussian_width.value(),
                    temperature=self.control_panel.desw_temperature.value(),
                    max_iter=self.control_panel.desw_max_iter.value(),
                )
        except Exception as e:
            QMessageBox.warning(self, "算法初始化错误", f"创建算法实例失败: {e}")
            return None

    def _toggle_play(self):
        """开始/暂停播放"""
        if self.is_running:
            # 暂停
            self.timer.stop()
            self.is_running = False
            self.control_panel.play_btn.setText("继续")
            self.control_panel.play_btn.setObjectName("playBtn")
            self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
        else:
            # 开始
            if self.algorithm is None:
                self.algorithm = self._create_algorithm()
                if self.algorithm is None:
                    if self._needs_two_endpoints():
                        QMessageBox.warning(self, "提示", "请先设置初始点和终态（在势能面上点击或手动输入）")
                    else:
                        QMessageBox.warning(self, "提示", "请先设置初始点（在势能面上点击或手动输入）")
                    return
                # 对比算法
                if self.control_panel.compare_check.isChecked():
                    compare_type = self.control_panel.compare_algo_combo.currentText()
                    # 如果有保存的对比参数，临时恢复后创建算法
                    saved_params = None
                    if hasattr(self, '_compare_params') and self._compare_params:
                        saved_params = self._capture_algo_params()
                        self._restore_algo_params(self._compare_params)
                    self.compare_algorithm = self._create_algorithm(compare_type)
                    # 恢复主算法的参数
                    if saved_params is not None:
                        self._restore_algo_params(saved_params)
                    self.compare_trajectory = []
                    self._cmp_converge_notified = False  # 重置收敛提示标志
                    self._main_converge_notified = False

            if self.algorithm.is_converged:
                QMessageBox.information(self, "提示", "算法已收敛，请重置后再开始")
                return

            speed = self.control_panel.get_speed_ms()
            self.timer.start(speed)
            self.is_running = True
            self.control_panel.play_btn.setText("暂停")
            self.control_panel.play_btn.setObjectName("pauseBtn")
            self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())

    def _timer_step(self):
        """定时器回调：执行一步"""
        # 主算法和对比算法都收敛时才停止
        main_converged = self.algorithm is None or self.algorithm.is_converged
        cmp_converged = True
        if self.control_panel.compare_check.isChecked() and self.compare_algorithm is not None:
            cmp_converged = self.compare_algorithm.is_converged

        if main_converged and cmp_converged:
            self.timer.stop()
            self.is_running = False
            self.control_panel.play_btn.setText("开始")
            self.control_panel.play_btn.setObjectName("playBtn")
            self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
            if self.algorithm and self.algorithm.is_converged:
                self.info_panel.update_result(
                    self.trajectory,
                    self.control_panel.get_algo_type(),
                    self.pes,
                )
                # 对比结果
                if self.control_panel.compare_check.isChecked() and self.compare_trajectory:
                    self._update_compare_results()
            # 对比模式下：若尚未提示过收敛，补充提示（避免定时器先停止导致_do_step未弹框）
            if self.control_panel.compare_check.isChecked() and self.compare_algorithm is not None:
                cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
                cmp_iter = getattr(self.compare_algorithm, '_iter_count', len(self.compare_trajectory))
                main_algo_type = self.control_panel.get_algo_type()
                main_iter = len(self.trajectory)
                msg_parts = []
                if self.algorithm is not None and self.algorithm.is_converged and not getattr(self, '_main_converge_notified', False):
                    msg_parts.append(f"主算法({main_algo_type})在 {main_iter} 步后收敛！")
                    self._main_converge_notified = True
                if self.compare_algorithm.is_converged and not getattr(self, '_cmp_converge_notified', False):
                    msg_parts.append(f"对比算法({cmp_algo_type})在 {cmp_iter} 步后收敛！")
                    self._cmp_converge_notified = True
                if msg_parts:
                    QMessageBox.information(self, "对比结束", "\n".join(msg_parts))
            return

        # 检查主算法是否超过最大迭代次数
        main_max_reached = False
        if not main_converged and self._check_max_iter_reached():
            main_max_reached = True
        # 检查对比算法是否超过最大迭代次数
        cmp_max_reached = False
        if (self.control_panel.compare_check.isChecked() 
                and self.compare_algorithm is not None 
                and not cmp_converged):
            cmp_max_iter = getattr(self.compare_algorithm, 'max_iter', None)
            if cmp_max_iter is not None:
                cmp_iter = getattr(self.compare_algorithm, '_iter_count', len(self.compare_trajectory))
                if cmp_iter >= cmp_max_iter:
                    cmp_max_reached = True

        # 两个算法都达到上限或收敛时停止
        if (main_converged or main_max_reached) and (cmp_converged or cmp_max_reached):
            self.timer.stop()
            self.is_running = False
            self.control_panel.play_btn.setText("开始")
            self.control_panel.play_btn.setObjectName("playBtn")
            self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
            if self.control_panel.compare_check.isChecked() and self.compare_trajectory:
                self._update_compare_results()
            # 对比模式下：结束时弹框提示（无论收敛与否）
            if self.control_panel.compare_check.isChecked() and self.compare_algorithm is not None:
                cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
                cmp_iter = getattr(self.compare_algorithm, '_iter_count', len(self.compare_trajectory))
                main_algo_type = self.control_panel.get_algo_type()
                main_iter = len(self.trajectory)
                msg_parts = []
                # 主算法状态
                if self.algorithm is not None and not getattr(self, '_main_converge_notified', False):
                    if self.algorithm.is_converged:
                        msg_parts.append(f"主算法({main_algo_type})在 {main_iter} 步后收敛！")
                    else:
                        msg_parts.append(f"主算法({main_algo_type})达到最大迭代次数 {main_iter}，未收敛！")
                    self._main_converge_notified = True
                # 对比算法状态
                if not getattr(self, '_cmp_converge_notified', False):
                    if self.compare_algorithm.is_converged:
                        msg_parts.append(f"对比算法({cmp_algo_type})在 {cmp_iter} 步后收敛！")
                    else:
                        msg_parts.append(f"对比算法({cmp_algo_type})达到最大迭代次数 {cmp_iter}，未收敛！")
                    self._cmp_converge_notified = True
                if msg_parts:
                    QMessageBox.information(self, "对比结束", "\n".join(msg_parts))
            return

        self._do_step()

    def _single_step(self):
        """单步执行"""
        if self.algorithm is None:
            self.algorithm = self._create_algorithm()
            if self.algorithm is None:
                if self._needs_two_endpoints():
                    QMessageBox.warning(self, "提示", "请先设置初始点和终态")
                else:
                    QMessageBox.warning(self, "提示", "请先设置初始点")
                return
            # 对比算法
            if self.control_panel.compare_check.isChecked():
                compare_type = self.control_panel.compare_algo_combo.currentText()
                saved_params = None
                if hasattr(self, '_compare_params') and self._compare_params:
                    saved_params = self._capture_algo_params()
                    self._restore_algo_params(self._compare_params)
                self.compare_algorithm = self._create_algorithm(compare_type)
                if saved_params is not None:
                    self._restore_algo_params(saved_params)
                self.compare_trajectory = []
                self._cmp_converge_notified = False  # 重置收敛提示标志
                self._main_converge_notified = False

        # 对比模式下：主算法和对比算法都收敛才提示
        main_converged = self.algorithm.is_converged
        cmp_converged = True
        if self.control_panel.compare_check.isChecked() and self.compare_algorithm is not None:
            cmp_converged = self.compare_algorithm.is_converged
        if main_converged and cmp_converged:
            QMessageBox.information(self, "提示", "算法已收敛")
            return

        # 检查主算法最大迭代次数（对比模式下不阻止，仅主算法达到上限时）
        if not main_converged and self._check_max_iter_reached():
            if cmp_converged:
                return

        self._do_step()

    def _check_max_iter_reached(self):
        """检查算法是否已达到最大迭代次数"""
        if self.algorithm is None:
            return False
        max_iter = getattr(self.algorithm, 'max_iter', None)
        if max_iter is None:
            return False
        # 获取当前迭代次数
        iter_count = getattr(self.algorithm, '_iter_count', len(self.trajectory))
        if iter_count >= max_iter:
            self.info_panel.update_result(
                self.trajectory,
                self.control_panel.get_algo_type(),
                self.pes,
            )
            QMessageBox.information(self, "达到上限", f"已达到最大迭代次数 {max_iter}，算法未收敛")
            return True
        return False

    def _do_step(self):
        """执行一步算法并更新显示"""
        state = None
        # 主算法步进（若未收敛）
        if self.algorithm is not None and not self.algorithm.is_converged:
            try:
                state = self.algorithm.step()
                self.trajectory.append(state)
            except Exception as e:
                self.timer.stop()
                self.is_running = False
                self.control_panel.play_btn.setText("开始")
                self.control_panel.play_btn.setObjectName("playBtn")
                self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
                QMessageBox.warning(self, "计算错误", f"算法执行出错: {e}")
                return

        # 对比算法步进（若未收敛）
        if self.control_panel.compare_check.isChecked() and self.compare_algorithm is not None:
            if not self.compare_algorithm.is_converged:
                try:
                    compare_state = self.compare_algorithm.step()
                    self.compare_trajectory.append(compare_state)
                except Exception:
                    pass

        # 更新偏置势可视化（SSW/MetaDynamics等算法会修改势能面）
        if self.algorithm is not None:
            bias_func = getattr(self.algorithm, 'get_bias_function', lambda: None)()
            self.canvas.set_bias_function(bias_func)

        # 更新信息面板（使用最新的主算法状态）
        algo_type = self.control_panel.get_algo_type()
        is_compare_mode = self.control_panel.compare_check.isChecked()
        if state is not None:
            self.info_panel.update_physics(state, algo_type)
        elif self.trajectory:
            self.info_panel.update_physics(self.trajectory[-1], algo_type)

        # 更新可视化
        self._update_view()

        # 检查收敛：主算法和对比算法分别独立提示
        main_just_converged = self.algorithm is not None and self.algorithm.is_converged and state is not None
        cmp_just_converged = False
        cmp_converged = True
        if is_compare_mode and self.compare_algorithm is not None:
            cmp_converged = self.compare_algorithm.is_converged
            # 对比算法刚收敛：本次步进前未收敛（标志位），步进后收敛
            # 使用标志位避免收敛后每个定时器周期都重复弹框
            if cmp_converged and not getattr(self, '_cmp_converge_notified', False):
                cmp_just_converged = True
                self._cmp_converge_notified = True

        if main_just_converged:
            # 主算法收敛，更新结果
            self.canvas.set_bias_function(None)
            self.info_panel.update_result(self.trajectory, algo_type, self.pes)
            if is_compare_mode and self.compare_trajectory:
                self._update_compare_results()
            self._main_converge_notified = True  # 标记主算法已提示收敛
            # 若对比算法也已收敛，停止定时器
            if cmp_converged:
                self.timer.stop()
                self.is_running = False
                self.control_panel.play_btn.setText("开始")
                self.control_panel.play_btn.setObjectName("playBtn")
                self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
            # 对比模式下：若对比算法同时收敛，合并提示
            if is_compare_mode and cmp_converged and not getattr(self, '_cmp_converge_notified', False):
                cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
                cmp_iter = getattr(self.compare_algorithm, '_iter_count', len(self.compare_trajectory))
                self._cmp_converge_notified = True
                QMessageBox.information(self, "对比完成",
                    f"主算法({algo_type})在 {state.get('iteration', 0)} 步后收敛！\n"
                    f"对比算法({cmp_algo_type})在 {cmp_iter} 步后收敛！")
            else:
                QMessageBox.information(self, "主算法收敛", f"主算法({algo_type})在 {state.get('iteration', 0)} 步后收敛！")
        elif cmp_just_converged and is_compare_mode:
            # 对比算法刚收敛（主算法未收敛），更新对比结果
            cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
            cmp_iter = getattr(self.compare_algorithm, '_iter_count', len(self.compare_trajectory))
            if self.compare_trajectory:
                self._update_compare_results()
            QMessageBox.information(self, "对比算法收敛", f"对比算法({cmp_algo_type})在 {cmp_iter} 步后收敛！")

    def _back_step(self):
        """回退一步"""
        if not self.trajectory:
            return

        # 移除最后一步（主算法）
        self.trajectory.pop()

        # 对比算法也回退一步（保持同步）
        if self.control_panel.compare_check.isChecked() and self.compare_trajectory:
            self.compare_trajectory.pop()
            # 重建对比算法并重放
            if self.compare_algorithm is not None:
                cmp_algo_type = self.control_panel.compare_algo_combo.currentText()
                saved_params = None
                if hasattr(self, '_compare_params') and self._compare_params:
                    saved_params = self._capture_algo_params()
                    self._restore_algo_params(self._compare_params)
                self.compare_algorithm = self._create_algorithm(cmp_algo_type)
                if saved_params is not None:
                    self._restore_algo_params(saved_params)
                if self.compare_algorithm is not None:
                    for _ in range(len(self.compare_trajectory)):
                        self.compare_algorithm.step()

        # 恢复主算法状态
        if self.algorithm is not None:
            # 重建算法并重放到上一步
            algo_type = self.control_panel.get_algo_type()
            self.algorithm = self._create_algorithm(algo_type)
            if self.algorithm is not None:
                for _ in range(len(self.trajectory)):
                    self.algorithm.step()

        # 更新显示
        if self.trajectory:
            self.info_panel.update_physics(
                self.trajectory[-1],
                self.control_panel.get_algo_type(),
            )
        else:
            self.info_panel.reset()

        self._update_view()

    def _reset(self):
        """重置"""
        self._reset_state()
        self._update_view()
        self.info_panel.reset()
        # 清除对比结果表格
        self.compare_table_main.setVisible(False)
        self.compare_table_main.setRowCount(0)
        algo_type = self.control_panel.get_algo_type()
        self.info_panel.update_principle(algo_type)

    def _show_help(self):
        """显示使用说明对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("使用说明")
        dlg.resize(720, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # 深色背景，白色字体
        browser.setStyleSheet("QTextBrowser { background-color: #1a1b2e; color: #ffffff; }")
        browser.setHtml(self._get_help_html())
        layout.addWidget(browser)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        dlg.exec_()

    def _get_help_html(self):
        """返回使用说明的HTML内容"""
        return """
        <html><body style='font-family: Microsoft YaHei; font-size: 13px; line-height: 1.6; color: #ffffff; background-color: #1a1b2e;'>
        <h2 style='color: #00d4ff;'>势能面可视化工具 使用说明</h2>

        <h3 style='color: #4fc3f7;'>一、基本操作</h3>
        <ul>
        <li><b>选择势能面：</b>左上角下拉框选择内置势能面（Müller-Brown、双阱、Rosenbrock等），或选择"自定义"输入表达式。</li>
        <li><b>调整范围：</b>在"显示范围"区修改 x/y 的最小/最大值，点击"应用"按钮刷新。</li>
        <li><b>切换视图：</b>顶部按钮切换"等高线图"和"3D曲面图"。</li>
        <li><b>3D视图操作：</b>左键拖拽旋转，右键拖拽平移，滚轮缩放。</li>
        </ul>

        <h3 style='color: #4fc3f7;'>二、算法运行</h3>
        <ul>
        <li><b>选择算法：</b>左栏"算法选择"下拉框选择优化算法。</li>
        <li><b>设置初始点：</b>点击"设置"按钮，然后在势能面图上点击选择初始点。部分算法（NEB、Dimer等）需要两个点。</li>
        <li><b>调整参数：</b>在"算法参数"区调整步长、收敛阈值等参数。</li>
        <li><b>开始/暂停：</b>点击"开始"按钮启动算法，运行中变为"暂停"。</li>
        <li><b>单步执行：</b>点击"单步"按钮逐步执行算法。</li>
        <li><b>回退：</b>点击"回退一步"撤销上一步。</li>
        <li><b>重置：</b>点击"重置"清除轨迹重新开始。</li>
        </ul>

        <h3 style='color: #4fc3f7;'>三、点标注说明</h3>
        <ul>
        <li><span style='color:#66bb6a;'>●</span> <b>绿色圆点</b>：势能面的极小值点</li>
        <li><span style='color:#ef5350;'>▲</span> <b>红色三角</b>：势能面的鞍点</li>
        <li><span style='color:#2196F3;'>■</span> <b>蓝色方块</b>：算法起点</li>
        <li><span style='color:#FF9800;'>●</span> <b>橙色小点</b>：算法轨迹中间点</li>
        <li><span style='color:#F44336;'>●</span> <b>红色圆点</b>：算法终点（收敛点）</li>
        <li><span style='color:#00BCD4;'>●</span> <b>青色点</b>：NEB弹性带图像</li>
        <li><span style='color:#9C27B0;'>●</span> <b>紫色点</b>：群体算法种群粒子</li>
        </ul>
        <p>右上角半透明框显示当前图中的点类型图例（等高线图和3D图均有）。</p>

        <h3 style='color: #4fc3f7;'>四、导出功能</h3>
        <ul>
        <li><b>导出CSV：</b>导出轨迹数据（坐标、能量、梯度等）。</li>
        <li><b>导出PNG：</b>导出当前视图为图片。</li>
        <li><b>导出GIF：</b>导出算法搜索过程的动画（等高线图+两个3D视角）。</li>
        </ul>

        <h3 style='color: #4fc3f7;'>五、方法对比</h3>
        <ul>
        <li>勾选"启用方法对比"，选择第二种算法。</li>
        <li>点击"编辑参数"可设置第二种算法的参数，设置完成后点击"完成参数"返回。</li>
        <li>启用后图形区分为左右两部分，分别显示两种算法的运行过程。</li>
        <li>图形与下栏之间显示两种方法的对比结果（收敛步数、最终能量等）。</li>
        <li>导出GIF时同时导出两种方法的动画。</li>
        </ul>

        <h3 style='color: #4fc3f7;'>六、下栏信息</h3>
        <ul>
        <li><b>实时物理量：</b>当前坐标、能量、梯度模长、Hessian本征值等。</li>
        <li><b>结果统计：</b>收敛步数、最终能量、收敛类型。</li>
        <li><b>势能面公式：</b>当前势能面的数学表达式。</li>
        <li><b>算法原理：</b>当前算法的原理说明。</li>
        </ul>

        <hr>
        <p style='color:#aaaaaa; font-size:11px;'>提示：字体大小会根据窗口大小自动调整。如遇显示问题，可尝试调整窗口大小。</p>
        </body></html>
        """

    def _reset_state(self):
        """重置内部状态"""
        self.timer.stop()
        self.is_running = False
        self.control_panel.play_btn.setText("开始")
        self.control_panel.play_btn.setObjectName("playBtn")
        self.control_panel.play_btn.setStyle(self.control_panel.play_btn.style())
        self.algorithm = None
        self.compare_algorithm = None
        self.trajectory = []
        self.compare_trajectory = []
        self._cmp_converge_notified = False  # 重置收敛提示标志
        self._main_converge_notified = False
        self.start_pos = None
        self.end_pos = None
        self.click_mode = 'start'
        self._point_setting_mode = False
        # 清除偏置势
        self.canvas.set_bias_function(None)
        self.control_panel.set_point_btn.setEnabled(True)
        self.control_panel.finish_point_btn.setEnabled(False)
        self.control_panel.init_x1.clear()
        self.control_panel.init_y1.clear()
        self.control_panel.init_x2.clear()
        self.control_panel.init_y2.clear()
        self.control_panel.init_hint.setText("点击'设置'开始选择初始点")

    def _update_compare_results(self):
        """更新对比结果"""
        results = []
        algo_type = self.control_panel.get_algo_type()
        compare_type = self.control_panel.compare_algo_combo.currentText()

        # 主算法结果
        if self.trajectory:
            last = self.trajectory[-1]
            results.append({
                'method': algo_type,
                'steps': len(self.trajectory),
                'energy': last.get('energy', 0),
                'type': self._get_converge_type(last),
            })

        # 对比算法结果
        if self.compare_trajectory:
            last = self.compare_trajectory[-1]
            results.append({
                'method': compare_type,
                'steps': len(self.compare_trajectory),
                'energy': last.get('energy', 0),
                'type': self._get_converge_type(last),
            })

        # 同时更新InfoPanel的表格和主对比面板的表格
        self.info_panel.show_compare_table(results)
        self._show_main_compare_table(results)

    def _show_main_compare_table(self, results):
        """在主对比面板显示对比结果表格"""
        self.compare_table_main.setVisible(True)
        # 先清空旧内容
        self.compare_table_main.clearContents()
        self.compare_table_main.setRowCount(len(results))
        self.compare_table_main.setColumnCount(4)
        self.compare_table_main.setHorizontalHeaderLabels(
            ["方法", "收敛步数", "最终能量", "收敛类型"]
        )
        for i, r in enumerate(results):
            self.compare_table_main.setItem(i, 0, QTableWidgetItem(r.get('method', '')))
            self.compare_table_main.setItem(i, 1, QTableWidgetItem(str(r.get('steps', ''))))
            self.compare_table_main.setItem(i, 2, QTableWidgetItem(f"{r.get('energy', 0):.6f}"))
            self.compare_table_main.setItem(i, 3, QTableWidgetItem(r.get('type', '')))
        self.compare_table_main.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 强制刷新
        self.compare_table_main.update()

    def _on_compare_algo_changed(self):
        """对比算法类型切换时重置对比状态"""
        # 仅在非编辑参数模式下处理（编辑模式下algo_combo会触发此信号）
        if getattr(self, '_editing_compare_params', False):
            return
        # 重置对比算法状态
        self.compare_algorithm = None
        self.compare_trajectory = []
        self._cmp_converge_notified = False  # 重置收敛提示标志
        self._main_converge_notified = False
        self.compare_table_main.setVisible(False)
        self.compare_table_main.setRowCount(0)
        # 更新视图
        if self.pes is not None:
            self._update_view()

    def _on_compare_mode_changed(self):
        """方法对比模式切换"""
        enabled = self.control_panel.compare_check.isChecked()
        self.canvas2.setVisible(enabled)
        self.compare_panel.setVisible(enabled)
        # 重置对比算法状态
        if not enabled:
            self.compare_algorithm = None
            self.compare_trajectory = []
            self.compare_table_main.setVisible(False)
            # 如果正在编辑参数，恢复算法选择下拉框
            cp = self.control_panel
            if getattr(self, '_editing_compare_params', False):
                self._editing_compare_params = False
                if hasattr(self, '_saved_algo_index'):
                    cp.algo_combo.blockSignals(True)
                    cp.algo_combo.setCurrentIndex(self._saved_algo_index)
                    cp._update_algo_params()
                    cp.algo_combo.blockSignals(False)
                cp.algo_combo.setEnabled(True)
                cp.compare_algo_combo.setEnabled(True)
                cp.play_btn.setEnabled(True)
                cp.step_btn.setEnabled(True)
                cp.compare_edit_params_btn.setText("编辑参数")
                cp.compare_edit_params_btn.setStyleSheet("")
        # 更新视图
        if self.pes is not None:
            self.canvas2.compute_grid(self.pes)
            self._update_view()

    def _toggle_compare_params_edit(self):
        """切换参数面板编辑主算法或对比算法的参数"""
        cp = self.control_panel
        if cp.compare_edit_params_btn.text() == "编辑参数":
            # 切换到对比算法的参数显示
            self._editing_compare_params = True
            self._saved_algo_index = cp.algo_combo.currentIndex()
            cmp_type = cp.compare_algo_combo.currentText()
            # 找到对比算法在algo_combo中的索引
            idx = cp.algo_combo.findText(cmp_type)
            if idx >= 0:
                cp.algo_combo.blockSignals(True)
                cp.algo_combo.setCurrentIndex(idx)
                cp._update_algo_params()
                cp.algo_combo.blockSignals(False)
            # 禁用算法选择下拉框，避免用户手动切换产生不一致
            cp.algo_combo.setEnabled(False)
            cp.compare_algo_combo.setEnabled(False)
            # 禁用播放控制按钮，避免在编辑参数时启动算法
            cp.play_btn.setEnabled(False)
            cp.step_btn.setEnabled(False)
            cp.compare_edit_params_btn.setText("完成参数")
            cp.compare_edit_params_btn.setStyleSheet("background-color: #FF9800; color: white;")
        else:
            # 保存对比算法的参数值
            self._compare_params = self._capture_algo_params()
            # 返回主算法的参数显示
            self._editing_compare_params = False
            cp.algo_combo.blockSignals(True)
            cp.algo_combo.setCurrentIndex(self._saved_algo_index)
            cp._update_algo_params()
            cp.algo_combo.blockSignals(False)
            # 恢复算法选择下拉框和播放控制按钮
            cp.algo_combo.setEnabled(True)
            cp.compare_algo_combo.setEnabled(True)
            cp.play_btn.setEnabled(True)
            cp.step_btn.setEnabled(True)
            cp.compare_edit_params_btn.setText("编辑参数")
            cp.compare_edit_params_btn.setStyleSheet("")

    def _capture_algo_params(self):
        """捕获当前所有算法参数spinbox的值，返回字典"""
        cp = self.control_panel
        params = {}
        for name in ['newton_threshold', 'newton_max_iter',
                     'dimer_delta_r', 'dimer_threshold', 'dimer_max_iter', 'dimer_trans_step',
                     'neb_n_images', 'neb_spring_k', 'neb_threshold', 'neb_max_iter',
                     'sd_step_size', 'sd_threshold', 'sd_max_iter',
                     'bh_step_size', 'bh_temperature', 'bh_max_iter',
                     'mc_step_size', 'mc_temperature', 'mc_max_iter',
                     'sSW_step_size', 'sSW_gaussian_height', 'sSW_gaussian_width', 'sSW_temperature', 'sSW_max_iter',
                     'meta_gaussian_height', 'meta_gaussian_width', 'meta_max_iter',
                     'mh_kinetic', 'mh_max_iter',
                     'ga_pop_size', 'ga_mutation_rate', 'ga_max_iter', 'ga_search_range',
                     'pso_n_particles', 'pso_w', 'pso_c1', 'pso_c2', 'pso_max_iter',
                     'abc_n_bees', 'abc_limit', 'abc_max_iter', 'abc_search_range',
                     'us_n_windows', 'us_spring_k', 'us_mc_step_size', 'us_mc_temperature', 'us_steps_per_window', 'us_max_iter',
                     'abf_n_bins', 'abf_mc_step_size', 'abf_mc_temperature', 'abf_max_iter',
                     'cbd_delta_r', 'cbd_threshold', 'cbd_trans_step', 'cbd_max_iter',
                     'desw_step_size', 'desw_gaussian_height', 'desw_gaussian_width', 'desw_temperature', 'desw_max_iter']:
            spin = getattr(cp, name, None)
            if spin is not None:
                try:
                    params[name] = spin.value()
                except Exception:
                    pass
        return params

    def _restore_algo_params(self, params):
        """从字典恢复算法参数spinbox的值"""
        cp = self.control_panel
        for name, value in params.items():
            spin = getattr(cp, name, None)
            if spin is not None:
                try:
                    spin.setValue(value)
                except Exception:
                    pass

    def _get_converge_type(self, state):
        """获取收敛类型描述"""
        if 'hessian_eigenvalues' in state:
            ev = state['hessian_eigenvalues']
            if np.all(ev > 0):
                return "极小值"
            elif np.sum(ev < 0) == 1:
                return "过渡态"
            else:
                return "发散"
        elif 'curvature' in state:
            return "过渡态" if state.get('curvature', 0) < 0 else "未找到鞍点"
        else:
            return "路径优化"

    def _export_csv(self):
        """导出轨迹数据为CSV"""
        if not self.trajectory:
            QMessageBox.warning(self, "提示", "没有轨迹数据可导出")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出轨迹数据", "trajectory.csv", "CSV文件 (*.csv)"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 写入表头
                if 'position' in self.trajectory[0]:
                    writer.writerow(['迭代步', 'x', 'y', '能量', '梯度模长',
                                     'Hessian λ₁', 'Hessian λ₂'])
                    for s in self.trajectory:
                        pos = s['position']
                        ev = s.get('hessian_eigenvalues', [0, 0])
                        writer.writerow([
                            s['iteration'], f"{pos[0]:.6f}", f"{pos[1]:.6f}",
                            f"{s['energy']:.6f}", f"{s['gradient_norm']:.6f}",
                            f"{ev[0]:.6f}", f"{ev[1]:.6f}",
                        ])
                elif 'center' in self.trajectory[0]:
                    writer.writerow(['迭代步', 'x', 'y', '能量', '梯度模长',
                                     '曲率', '双子方向x', '双子方向y',
                                     '平行力', '垂直力'])
                    for s in self.trajectory:
                        c = s['center']
                        d = s.get('direction', [0, 0])
                        writer.writerow([
                            s['iteration'], f"{c[0]:.6f}", f"{c[1]:.6f}",
                            f"{s['energy']:.6f}", f"{s['gradient_norm']:.6f}",
                            f"{s.get('curvature', 0):.6f}",
                            f"{d[0]:.6f}", f"{d[1]:.6f}",
                            f"{s.get('f_parallel', 0):.6f}",
                            f"{s.get('f_perp', 0):.6f}",
                        ])
                else:
                    writer.writerow(['迭代步', '最大力', '最高能量'])
                    for s in self.trajectory:
                        writer.writerow([
                            s['iteration'],
                            f"{s.get('max_force', 0):.6f}",
                            f"{np.max(s.get('energies', [0])):.6f}",
                        ])

            QMessageBox.information(self, "导出成功", f"轨迹数据已保存到:\n{filepath}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"保存CSV文件失败: {e}")

    def _export_png(self):
        """导出势能面图片为PNG"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出势能面图片", "pes_plot.png", "PNG图片 (*.png)"
        )
        if not filepath:
            return
        try:
            self.canvas.save_png(filepath)
            QMessageBox.information(self, "导出成功", f"图片已保存到:\n{filepath}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"保存PNG文件失败: {e}")

    def _export_gif(self):
        """导出搜索过程GIF - 三视图（等高线+两个3D视角）拼接"""
        if not self.trajectory:
            QMessageBox.warning(self, "提示", "没有搜索轨迹可导出，请先运行算法")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出搜索过程GIF", "search_process.gif", "GIF动画 (*.gif)"
        )
        if not filepath:
            return

        try:
            from PIL import Image

            # 保存当前布局状态
            old_mode = self.canvas.mode
            old_layout = self.canvas._current_layout
            old_user_xlim = self.canvas._user_xlim
            old_user_ylim = self.canvas._user_ylim
            old_user_dist = self.canvas._user_dist
            old_cam_state = self.canvas._save_camera_state()

            # 计算完整轨迹的范围
            all_points = []
            for s in self.trajectory:
                if 'position' in s:
                    all_points.append(s['position'])
                elif 'center' in s:
                    all_points.append(s['center'])
                if 'images' in s:
                    for img in s['images']:
                        all_points.append(img)
            if self.start_pos is not None:
                all_points.append(self.start_pos)
            if self.end_pos is not None:
                all_points.append(self.end_pos)

            gif_xlim = None
            gif_ylim = None
            if all_points:
                all_points = np.array(all_points)
                traj_xmin, traj_xmax = all_points[:, 0].min(), all_points[:, 0].max()
                traj_ymin, traj_ymax = all_points[:, 1].min(), all_points[:, 1].max()
                xpad = max((traj_xmax - traj_xmin) * 0.2, 0.1)
                ypad = max((traj_ymax - traj_ymin) * 0.2, 0.1)
                gif_xlim = (traj_xmin - xpad, traj_xmax + xpad)
                gif_ylim = (traj_ymin - ypad, traj_ymax + ypad)

            # 强制重建布局
            self.canvas._current_layout = None

            # 是否为方法对比模式
            is_compare_mode = self.control_panel.compare_check.isChecked() and len(self.compare_trajectory) > 0
            if is_compare_mode:
                # 确保canvas2已计算网格
                if self.canvas2.Z is None:
                    self.canvas2.compute_grid(self.pes)
                self.canvas2._current_layout = None

            # 3D视角参数
            cam_angles = [
                {'elevation': 30, 'azimuth': 45},   # 默认斜视角
                {'elevation': 60, 'azimuth': -30},   # 高角度俯视
            ]

            def render_frame(partial_traj, start_pos, end_pos, dimer_info, neb_images, energy_profile, population_positions=None,
                             cmp_partial_traj=None, cmp_dimer_info=None, cmp_neb_images=None, cmp_energy_profile=None, cmp_pop_positions=None):
                """渲染一帧的三视图并拼接"""
                frame_images = []

                # 视图1：等高线图（主算法）
                self.canvas._current_layout = None
                self.canvas.draw_contour(
                    trajectory=partial_traj,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    dimer_info=dimer_info,
                    neb_images=neb_images,
                    energy_profile=energy_profile,
                    population_positions=population_positions,
                )
                if gif_xlim is not None:
                    self.canvas.set_view_range(gif_xlim, gif_ylim)
                buf = self.canvas.get_frame_image()
                frame_images.append(Image.open(buf).copy())
                buf.close()

                # 视图1b：等高线图（对比算法，如果启用）
                if is_compare_mode:
                    self.canvas2._current_layout = None
                    self.canvas2.draw_contour(
                        trajectory=cmp_partial_traj,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        dimer_info=cmp_dimer_info,
                        neb_images=cmp_neb_images,
                        energy_profile=cmp_energy_profile,
                        population_positions=cmp_pop_positions,
                    )
                    if gif_xlim is not None:
                        self.canvas2.set_view_range(gif_xlim, gif_ylim)
                    buf = self.canvas2.get_frame_image()
                    frame_images.append(Image.open(buf).copy())
                    buf.close()

                # 视图2和3：两个3D视角（主算法）
                for cam in cam_angles:
                    self.canvas._current_layout = None
                    self.canvas.draw_3d(
                        trajectory=partial_traj,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        neb_images=neb_images,
                        energy_profile=energy_profile,
                        population_positions=population_positions,
                    )
                    # 设置3D相机角度
                    try:
                        self.canvas._gl_widget.opts['elevation'] = cam['elevation']
                        self.canvas._gl_widget.opts['azimuth'] = cam['azimuth']
                        self.canvas._gl_widget.update()
                    except Exception:
                        pass
                    buf = self.canvas.get_frame_image()
                    frame_images.append(Image.open(buf).copy())
                    buf.close()

                # 视图2b和3b：两个3D视角（对比算法，如果启用）
                if is_compare_mode:
                    for cam in cam_angles:
                        self.canvas2._current_layout = None
                        self.canvas2.draw_3d(
                            trajectory=cmp_partial_traj,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            neb_images=cmp_neb_images,
                            energy_profile=cmp_energy_profile,
                            population_positions=cmp_pop_positions,
                        )
                        try:
                            self.canvas2._gl_widget.opts['elevation'] = cam['elevation']
                            self.canvas2._gl_widget.opts['azimuth'] = cam['azimuth']
                            self.canvas2._gl_widget.update()
                        except Exception:
                            pass
                        buf = self.canvas2.get_frame_image()
                        frame_images.append(Image.open(buf).copy())
                        buf.close()

                # 拼接三视图：水平排列
                widths = [img.width for img in frame_images]
                heights = [img.height for img in frame_images]
                target_h = min(heights)
                # 等比缩放到相同高度
                resized = []
                for img in frame_images:
                    ratio = target_h / img.height
                    new_w = int(img.width * ratio)
                    resized.append(img.resize((new_w, target_h), Image.LANCZOS))
                total_w = sum(r.width for r in resized)
                combined = Image.new('RGB', (total_w, target_h))
                x_offset = 0
                for r in resized:
                    combined.paste(r, (x_offset, 0))
                    x_offset += r.width
                return combined

            images = []

            # 准备通用参数
            start_pos = self.start_pos
            end_pos = self.end_pos if self._needs_two_endpoints() else None
            algo_type = self.control_panel.get_algo_type()
            cmp_algo_type = self.control_panel.compare_algo_combo.currentText() if is_compare_mode else None

            # 第一帧：只有初始点
            cmp_args = {}
            if is_compare_mode:
                cmp_args = dict(cmp_partial_traj=None, cmp_dimer_info=None, cmp_neb_images=None,
                                cmp_energy_profile=None, cmp_pop_positions=None)
            combined = render_frame(None, start_pos, end_pos, None, None, None, **cmp_args)
            images.append(combined)

            # 后续帧：逐步添加轨迹
            # 确定最大帧数（主算法和对比算法中的较大值）
            max_frames = len(self.trajectory)
            if is_compare_mode:
                max_frames = max(max_frames, len(self.compare_trajectory))

            for i in range(1, max_frames + 1):
                # 主算法轨迹（截断到可用长度）
                partial_traj = self.trajectory[:i] if i <= len(self.trajectory) else self.trajectory

                dimer_info = None
                if algo_type == "Dimer方法" and partial_traj:
                    last = partial_traj[-1]
                    dimer_info = {
                        'center': last.get('center', self.start_pos),
                        'direction': last.get('direction', np.array([1, 0])),
                        'delta_r': self.control_panel.dimer_delta_r.value(),
                    }
                neb_images = None
                energy_profile = None
                if self._is_neb_algo() and self.algorithm is not None:
                    if partial_traj:
                        last = partial_traj[-1]
                        if 'images' in last:
                            neb_images = [img.tolist() for img in last['images']]
                        if 'path_energy_profile' in last:
                            distances, _ = self.algorithm.get_energy_profile()
                            energy_profile = (distances, last['path_energy_profile'])

                # 群体算法种群位置
                pop_positions = None
                if partial_traj:
                    last = partial_traj[-1]
                    if 'population_positions' in last:
                        pop_positions = last['population_positions']
                    elif 'particle_positions' in last:
                        pop_positions = last['particle_positions']

                # 偏置势可视化（SSW/MetaDynamics）
                if self.algorithm is not None and hasattr(self.algorithm, 'get_bias_function'):
                    # 重建算法到第i步以获取正确的偏置状态
                    temp_algo = self._create_algorithm(algo_type)
                    if temp_algo is not None:
                        for _ in range(i):
                            temp_algo.step()
                        bias_func = temp_algo.get_bias_function()
                        self.canvas.set_bias_function(bias_func)

                # 准备对比算法数据
                cmp_args = {}
                if is_compare_mode:
                    cmp_partial_traj = self.compare_trajectory[:i] if i <= len(self.compare_trajectory) else self.compare_trajectory
                    cmp_dimer_info = None
                    if cmp_algo_type == "Dimer方法" and cmp_partial_traj:
                        last = cmp_partial_traj[-1]
                        cmp_dimer_info = {
                            'center': last.get('center', self.start_pos),
                            'direction': last.get('direction', np.array([1, 0])),
                            'delta_r': self.control_panel.dimer_delta_r.value(),
                        }
                    cmp_neb_images = None
                    cmp_energy_profile = None
                    if cmp_algo_type in ("NEB方法", "CI-NEB方法") and self.compare_algorithm is not None:
                        if cmp_partial_traj:
                            last = cmp_partial_traj[-1]
                            if 'images' in last:
                                cmp_neb_images = [img.tolist() for img in last['images']]
                            if 'path_energy_profile' in last:
                                distances, _ = self.compare_algorithm.get_energy_profile()
                                cmp_energy_profile = (distances, last['path_energy_profile'])
                    cmp_pop_positions = None
                    if cmp_partial_traj:
                        last = cmp_partial_traj[-1]
                        if 'population_positions' in last:
                            cmp_pop_positions = last['population_positions']
                        elif 'particle_positions' in last:
                            cmp_pop_positions = last['particle_positions']
                    cmp_args = dict(
                        cmp_partial_traj=cmp_partial_traj, cmp_dimer_info=cmp_dimer_info,
                        cmp_neb_images=cmp_neb_images, cmp_energy_profile=cmp_energy_profile,
                        cmp_pop_positions=cmp_pop_positions)

                combined = render_frame(
                    partial_traj, start_pos, end_pos,
                    dimer_info, neb_images, energy_profile, pop_positions, **cmp_args)
                images.append(combined)

            # 恢复布局状态
            self.canvas.mode = old_mode
            self.canvas._current_layout = None
            self.canvas._cbar_ax = None
            self.canvas._user_xlim = old_user_xlim
            self.canvas._user_ylim = old_user_ylim
            self.canvas._user_dist = old_user_dist
            self.canvas._restore_camera_state(old_cam_state)
            # 恢复偏置势状态
            if self.algorithm is not None and hasattr(self.algorithm, 'get_bias_function'):
                self.canvas.set_bias_function(self.algorithm.get_bias_function())
            else:
                self.canvas.set_bias_function(None)
            self._update_view()

            if images:
                images[0].save(
                    filepath,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(self.control_panel.get_speed_ms()),
                    loop=0,
                )
                for img in images:
                    img.close()
                QMessageBox.information(self, "导出成功", f"GIF已保存到:\n{filepath}")
            else:
                QMessageBox.warning(self, "导出失败", "没有可用的帧数据")
        except ImportError:
            QMessageBox.warning(
                self, "缺少依赖",
                "导出GIF需要Pillow库，请安装: pip install Pillow"
            )
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"保存GIF文件失败: {e}")


# ======================== 主函数 ========================

def main():
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle('Fusion')

    # 应用QSS样式表
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # 设置暗色科技主题
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 27, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(34, 36, 58))
    palette.setColor(QPalette.AlternateBase, QColor(42, 45, 72))
    palette.setColor(QPalette.ToolTipBase, QColor(42, 45, 72))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(42, 45, 72))
    palette.setColor(QPalette.ButtonText, QColor(200, 204, 224))
    palette.setColor(QPalette.Highlight, QColor(9, 132, 227))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(100, 100, 120))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 120))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
