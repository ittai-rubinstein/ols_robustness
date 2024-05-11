from pathlib import Path
import shutil
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange  # Import tqdm for progress bars
import numpy as np
import os
CURRENT_FILE = Path(__file__).resolve()
RESULTS_PATH = CURRENT_FILE.parent / "results" / "figures" / "cs_vs_ra"
CS_OUTPUT_DIR = RESULTS_PATH / "covariance_shift"
RA_OUTPUT_DIR = RESULTS_PATH / "residual_attack"

os.makedirs(CS_OUTPUT_DIR, exist_ok=True)
os.makedirs(RA_OUTPUT_DIR, exist_ok=True)

DPI = 500
JUMP = 2
FIGSIZE = (9, 7)
TIME_IN_MS = 5000
FONTSIZE = 'xx-large'
YLIM = (-11, 11)
XLIM = (-11, 11)
LOOP_WAIT = 500


def fit_to_linear_growth(input_array):
    n = len(input_array)
    indices = np.arange(n)  # Array of indices

    # Fit a polynomial of degree 1 (line) to the input data
    coefficients = np.polyfit(indices, input_array, deg=1)

    # Generate the array that grows linearly with the index
    linear_fit = np.polyval(coefficients, indices)

    return linear_fit


def fit_to_inverse_func(input_array):
    n = len(input_array)
    base = input_array[0]
    cap = input_array[-1]
    idcs = np.arange(n)
    return np.clip(base + ((cap - base) / (n - idcs)), base, cap)


def plot_samples_with_fit(ax: plt.Axes, samples: np.ndarray, label: str = "", alpha: float = 1.0):
    # Scatter plot of the samples
    ax.scatter(samples[:, 0], samples[:, 1], color='blue', edgecolor='none', marker='.', alpha=alpha, label=label)

    # Perform least squares fit
    A = np.vstack([samples[:, 0], np.zeros(len(samples))]).T
    m, c = np.linalg.lstsq(A, samples[:, 1], rcond=None)[0]

    # Plot the least squares fit line
    x_vals = np.linspace(XLIM[0], XLIM[1], 100)
    ax.plot(x_vals, m * x_vals + c, 'b', label=f'{label} Fit', alpha=alpha)


def generate_frames_for_main_plot(dataset_1: np.ndarray, dataset_2: np.ndarray):
    # Function to generate frames for the main plot

    combined_dataset = np.concatenate((dataset_1, dataset_2), axis=0)
    temp_dir = tempfile.mkdtemp()

    frames = []

    for i in trange(0, len(dataset_2) + (1 * JUMP), JUMP):
        icap = min(i, len(dataset_2))
        fig, ax = plt.subplots(figsize=FIGSIZE)

        plot_samples_with_fit(ax, combined_dataset, label="Combined Dataset", alpha=0.5)
        ylim = ax.get_ylim()
        # ylim = (-11, 11)
        if i < len(dataset_2):
            modified_dataset_2 = np.concatenate((dataset_1, dataset_2[:-i - 1]))
        else:
            modified_dataset_2 = dataset_1

        plot_samples_with_fit(ax, modified_dataset_2, label="Subset", alpha=1.0)

        ax.set_xlabel('X', fontsize=FONTSIZE)
        ax.set_ylabel('Y', fontsize=FONTSIZE)
        ax.grid()
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        # ax.legend()

        plt.tight_layout()
        file_path = os.path.join(temp_dir, f"plot_{i}.png")
        plt.savefig(file_path, dpi=DPI)
        plt.close()

        frames.append(Image.open(file_path))

    return frames, temp_dir


def generate_small_plot(beta_values: np.ndarray, index: int):
    # Function to generate the smaller plot
    num_samples = len(beta_values) - 1
    fig_small, ax_small = plt.subplots(figsize=(4, 3))
    fit_beta_values = BETA_FIT(beta_values)
    ax_small.plot(range(num_samples + 1), fit_beta_values, '--r')
    if index < num_samples:
        ax_small.plot(range(0, index + 1, JUMP), beta_values[:index + 1:JUMP], '*r')
    else:
        ax_small.plot(
            np.concatenate((range(0, num_samples, JUMP), [num_samples])),
            np.concatenate((beta_values[:-1:JUMP], [beta_values[-1]])),
            '*r'
        )
    ax_small.set_xlabel('# Samples Removed', fontsize=FONTSIZE)
    ax_small.set_ylabel(r'$\beta$', fontsize=FONTSIZE)
    ax_small.set_xlim(0, num_samples + 1)
    ax_small.set_ylim(min(beta_values), max(beta_values))

    # Remove x and y ticks and labels
    ax_small.set_xticks([])
    ax_small.set_yticks([])
    ax_small.set_xticklabels([])
    ax_small.set_yticklabels([])

    temp_dir = tempfile.mkdtemp()
    small_plot_path = os.path.join(temp_dir, "small_plot.png")

    plt.savefig(small_plot_path, dpi=DPI)
    plt.close()

    small_plot = Image.open(small_plot_path)
    return small_plot, temp_dir


def generate_plots(dataset_1: np.ndarray, dataset_2: np.ndarray, output_dir: str):
    # Function to perform the whole process

    frames, temp_dir_main = generate_frames_for_main_plot(dataset_1, dataset_2)

    least_squares_results = []
    beta_values = []

    frames_with_small_plot = []

    for i in trange(0, len(dataset_2) + 1):
        if i < len(dataset_2):
            modified_dataset_2 = np.concatenate((dataset_1, dataset_2[:-i - 1]))
        else:
            modified_dataset_2 = dataset_1

        A_mod = np.vstack([modified_dataset_2[:, 0], np.zeros(len(modified_dataset_2))]).T
        m_mod, c_mod = np.linalg.lstsq(A_mod, modified_dataset_2[:, 1], rcond=None)[0]

        least_squares_results.append((m_mod, c_mod))
        if i <= len(dataset_2):
            beta_values.append(m_mod)

    for i in trange(0, len(dataset_2) + JUMP, JUMP):
        icap = min(i, len(dataset_2))
        small_plot, temp_dir_small = generate_small_plot(beta_values, i)
        frame = frames[icap // JUMP]
        frame_with_small_plot = Image.new('RGB', (frame.width, frame.height))
        frame_with_small_plot.paste(frame, (0, 0))
        frame_with_small_plot.paste(small_plot, (frame.width - small_plot.width, 0))
        frames_with_small_plot.append(frame_with_small_plot)

    frame_duration = int((TIME_IN_MS - LOOP_WAIT) / len(frames))

    if False:
        # Save the GIF without the small plot
        frames[0].save(os.path.join(output_dir, 'animated_plot.gif'),
                       format='GIF', append_images=frames[1:] + (frames[-1:] * (LOOP_WAIT // frame_duration)),
                       save_all=True, duration=frame_duration, loop=0)

        # Save the GIF with the small plot
        frames_with_small_plot[0].save(os.path.join(output_dir, 'animated_plot_with_small_plot.gif'),
                                       format='GIF', append_images=frames_with_small_plot[1:] + (
                        frames_with_small_plot[-1:] * (LOOP_WAIT // frame_duration)),
                                       save_all=True, duration=frame_duration, loop=0)

    frames_with_small_plot[-1].save(os.path.join(output_dir, 'last_frame.png'))

    shutil.rmtree(temp_dir_main)
    shutil.rmtree(temp_dir_small)


# Set random seed for reproducibility
np.random.seed(42)

# First dataset with 1000 points
num_points = 1000
x_values = -1 + (2*np.random.rand(num_points))
small_error = np.random.normal(0, 0.2, num_points)  # Small Gaussian error
y_values = x_values + small_error
# Storing the first dataset as an n by 2 NumPy array
dataset_1 = np.column_stack((x_values, y_values))

# Second dataset with 100 samples of (1, -1) or (-1, 1)
num_samples = 100
x_values = np.random.choice([-1, 1], size=(num_samples, 1))
FACTOR = 10
ERROR_SIZE=0.2
y_values = -10*x_values
dataset_2 = np.column_stack((
    x_values + np.random.normal(0, ERROR_SIZE , (num_samples, 1)),
    y_values + np.random.normal(0, ERROR_SIZE, (num_samples, 1))
))
dataset_2 = dataset_2

if False:
    BETA_FIT = fit_to_linear_growth
    generate_plots(dataset_1, dataset_2, RA_OUTPUT_DIR)


# Second dataset with 100 samples of (1, -1) or (-1, 1)
np.random.seed(1)
num_samples = 100
x_values = np.random.choice([-1, 1], size=(num_samples, 1))
FACTOR = 10
ERROR_SIZE=0.2
x_values *= FACTOR
y_values = -x_values
dataset_3 = np.column_stack((
    x_values + np.random.normal(0, ERROR_SIZE, (num_samples, 1)),
    y_values + np.random.normal(0, ERROR_SIZE , (num_samples, 1))
))



import numpy as np
from scipy.optimize import curve_fit


def fit_function_to_array(input_array):
    n = len(input_array)
    indices = np.arange(n)

    # Define initial guesses for the parameters
    initial_guess = (1.0, 1.0, 1.0)  # Initial values for (base_value, delta, weight, n)

    def func(i, base_value, delta, weight):
        return base_value + ((i * delta) / (weight + (n - i)))

    # Perform curve fitting
    popt, _ = curve_fit(func, indices, input_array, p0=initial_guess)

    # Extract fitted parameters
    fitted_base_value, fitted_delta, fitted_weight = popt

    # Compute the fitted array
    fitted_array = np.array([
        fitted_base_value + ((i * fitted_delta) / (fitted_weight + (n - i))) for i in range(n)
    ])

    return fitted_array


BETA_FIT = fit_function_to_array
generate_plots(dataset_1, dataset_3, CS_OUTPUT_DIR)