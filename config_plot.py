from matplotlib import pyplot as plt

##############################
# plot setting
# 自定义 IEEE 风格样式
ieee_style = {
    'font.family': 'sans-serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
}

# 保存默认样式
default_style = plt.rcParams.copy()

# 应用自定义 IEEE 风格
plt.rcParams.update(ieee_style)

DPI = 50
Latex_DPI = 300

##############################
# result setting
output_dir = "results"
