# Libraries
import matplotlib.pyplot as plt
import numpy as np

def plot_to_tensorboard(writer, fig, tag, step):
    """
    CGPT suggests to replace plot_to_tensorboard with this one.

    Args:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
    """
    writer.add_figure(tag=tag, figure=fig, global_step=step)
    plt.close(fig)

# def plot_to_tensorboard(writer, fig, name, step):
#     """
#     Takes a matplotlib figure handle and converts it using
#     canvas and string-casts to a numpy array that can be
#     visualized in TensorBoard using the add_image function

#     Stolen from https://martin-mundt.com/tensorboard-figures/

#     Args:
#         writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
#         fig (matplotlib.pyplot.fig): Matplotlib figure handle.
#     """

#     # Draw figure on canvas
#     fig.canvas.draw()

#     # Convert the figure to numpy array, read the pixel values and reshape the array
#     # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     img = np.asarray(fig.canvas.buffer_rgba())

#     # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
#     img = img / 255.0
#     # if your TensorFlow + TensorBoard version are >= 1.8
#     img = np.swapaxes(img, 0, 2)
#     img = np.swapaxes(img, 1, 2)

#     # Add figure in numpy "image" to TensorBoard writer
#     writer.add_image(name, img, step)
#     plt.close(fig)


def set_axis_white(ax):
    """Set an matplotlib.axes.Axes to use white colors"""

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
