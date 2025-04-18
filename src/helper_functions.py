# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set figure parameters for all plots
newParams = {'figure.figsize'  : (12, 6),  # Figure size
             'figure.dpi'      : 200,      # figure resolution
             'axes.titlesize'  : 20,       # fontsize of title
             'axes.labelsize'  : 11,       # fontsize of axes labels
             'axes.linewidth'  : 2,        # width of the figure box lines
             'lines.linewidth' : 2.5,        # width of the plotted lines
             'savefig.dpi'     : 200,      # resolution of a figured saved using plt.savefig(filename)
             'ytick.labelsize' : 11,       # fontsize of tick labels on y axis
             'xtick.labelsize' : 11,       # fontsize of tick labels on x axis
             'legend.fontsize' : 12,       # fontsize of labels in legend
             'legend.frameon'  : True,     # activate frame on lengend
            }
plt.rcParams.update(newParams) # Set new plotting parameters


def generate_time_series_psi_real(solution, num_time_steps):
    """Generates sample time-series graph data."""
    data = []
    for t in range(num_time_steps):
        data.append(np.column_stack((solution.x, np.real(solution.psi_total[t]))))
    return np.array(data)

def generate_time_series_psi_imag(solution, num_time_steps):
    """Generates sample time-series graph data."""
    data = []
    for t in range(num_time_steps):
        data.append(np.column_stack((solution.x, np.imag(solution.psi_total[t]))))
    return np.array(data)

def generate_time_series_psi_squared(solution, num_time_steps):
    """Generates sample time-series graph data."""
    data = []
    for t in range(num_time_steps):
        data.append(np.column_stack((solution.x, np.abs(solution.psi_total[t])**2)))
    return np.array(data)

def animate_psi(time_series_data, t_steps, filename, label, color):
    """Creates and saves an animation of time-series data."""
    fig, ax = plt.subplots()
    line, = ax.plot(time_series_data[0, :, 0], time_series_data[0, :, 1], color=color)  # Initial plot
    ax.set_xlim(np.min(time_series_data[:, :, 0]), np.max(time_series_data[:, :, 0]))
    ax.set_ylim(np.min(time_series_data[:, :, 1]), np.max(time_series_data[:, :, 1]))
    ax.set_title(filename)
    ax.set_xlabel("x")
    ax.set_ylabel(label)

    def animate(i):
        line.set_xdata(time_series_data[i, :, 0])
        line.set_ydata(time_series_data[i, :, 1])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=t_steps, interval=1, blit=True)
    ani.save(filename + ".mp4", writer='ffmpeg', fps=30)


def check_norm(psi, time, dx):
    integral = np.trapezoid(np.abs(psi) ** 2, dx=dx)
    return "P2 Norm at t {:.1f}: {:.9f} ".format(time, integral)

