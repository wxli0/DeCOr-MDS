import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import os

fig_path = os.path.join("./outputs/synthetic_dim10_S.pdf")
if not os.path.exists(fig_path):
    num_groups = [50, 100, 150, 200, 250]
    runtimes = [41.60270047187805, 64.90046525001526, 90.48285031318665, 119.87272953987122, 145.38690447807312]
    plt.plot(num_groups, runtimes, marker='.')
    plt.xticks(num_groups)
    plt.xlabel('number of nsimplices')
    plt.ylabel('runtime')
    plt.savefig(fig_path)

fig_path = os.path.join("./outputs/synthetic_dim10_N.pdf")
if not os.path.exists(fig_path):
    num_points = [500, 1000, 1500, 2000, 2500]
    num_points_fifth = [500**5, 1000**5, 1500**5, 2000**5, 2500**5]
    runtimes = [64.90046525001526, 87.05577659606934, 654.4376263618469, 1114.2744002342224, 1897.8766615390778]
    plt.plot(num_points_fifth, runtimes, marker='.')
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    plt.xticks(num_points_fifth)
    plt.xlabel(r'number of data points^5')
    plt.ylabel('runtime')
    plt.tight_layout
    plt.savefig(fig_path)
