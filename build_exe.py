"""打包脚本 - 通过Python子进程调用PyInstaller，避免PowerShell stderr问题"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 解决OpenMP重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

python_exe = sys.executable
cmd = [
    python_exe, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "PES_Visualizer",
    "--icon", "pesvisualizerlogo.png",
    "--add-data", "pesvisualizerlogo.png;.",
    "--hidden-import", "PyQt5",
    "--hidden-import", "PyQt5.QtCore",
    "--hidden-import", "PyQt5.QtGui",
    "--hidden-import", "PyQt5.QtWidgets",
    "--hidden-import", "pyqtgraph",
    "--hidden-import", "pyqtgraph.opengl",
    "--hidden-import", "pyqtgraph.graphicsItems",
    "--hidden-import", "numpy",
    "--hidden-import", "scipy",
    "--hidden-import", "scipy.optimize",
    "--hidden-import", "scipy.ndimage",
    "--hidden-import", "skimage",
    "--hidden-import", "skimage.measure",
    "--hidden-import", "PIL",
    "--hidden-import", "OpenGL",
    "--hidden-import", "OpenGL.GL",
    "--collect-data", "pyqtgraph",
    "--exclude-module", "matplotlib",
    "--exclude-module", "tkinter",
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "pandas",
    "--exclude-module", "torch",
    "--exclude-module", "tensorflow",
    "--exclude-module", "keras",
    "--exclude-module", "notebook",
    "--exclude-module", "sympy",
    "--exclude-module", "bokeh",
    "--exclude-module", "plotly",
    "--exclude-module", "networkx",
    "--exclude-module", "statsmodels",
    "--exclude-module", "seaborn",
    "--exclude-module", "scipy._lib.array_api_compat.torch",
    "main.py",
]

print("Starting PyInstaller...")
print(" ".join(cmd))

# 使用 Popen 实时输出，避免缓冲问题
import time
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1)
for line in proc.stdout:
    try:
        print(line.decode('utf-8', errors='replace'), end='', flush=True)
    except Exception:
        pass
proc.wait()
print(f"\nPyInstaller exited with code: {proc.returncode}")

if proc.returncode == 0:
    exe_path = os.path.join("dist", "PES_Visualizer", "PES_Visualizer.exe")
    if os.path.exists(exe_path):
        print(f"SUCCESS: {exe_path} created!")
    else:
        print("WARNING: PyInstaller exited 0 but exe not found")
else:
    print("FAILED: PyInstaller exited with error")
