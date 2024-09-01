import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter, MultipleLocator
# import seaborn as sns
# Apply the default theme
# sns.set_theme()
# sns.set(font_scale = 2)
# sns.set_style('white')

# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlepad=15)

# set tick width
plt.rcParams['xtick.major.size'] = 15 # default 3.5
plt.rcParams['xtick.major.width'] = 2 # default 0.8 

plt.rcParams['ytick.major.size'] = 14 # default 3.5
plt.rcParams['ytick.major.width'] = 2.4 # 0.8 

plt.rcParams['lines.linewidth'] = 2.5

DPI = 200
FIGSIZE = (12.5, 7)
DATE_TICKS = 5

markers = ['s', 'x', '+',  'h', 'D', '.', '^', '>', 'p', '<', '*', 'P', 'v']

def get_formatter(scale):
    return FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))

prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
obs_color = next(prop_cycle)["color"]
pred_color = next(prop_cycle)["color"]