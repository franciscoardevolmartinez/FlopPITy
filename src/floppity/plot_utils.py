import corner
import torch
import matplotlib.pyplot as plt

def plot_moment_evolution(distributions, end, compute_moments_fn, R=None, 
                            num_repeats=10, sample_size=10000):
    """
    Plot the evolution of moments across training rounds for a list of 
    multivariate distributions, with shaded error bars and tightly 
    stacked subplots.

    Parameters
    ----------
    distributions : list of torch.distributions.Distribution
        List of proposal distributions (one per training round).
    compute_moments_fn : function
        Function that returns a dict of per-dimension moment arrays and 
        their errors.
    R : optional
        Object with .parameters (e.g. from sbi) to label dimension names.
    """
    n_rounds = len(distributions)
    moments_all = [compute_moments_fn(d, num_repeats, sample_size) for d in distributions]

    moment_keys = ['mean', 'variance']#, 'skewness', 'kurtosis']
    error_keys = [f"{k}_error" for k in moment_keys]
    n_moments = len(moment_keys)

    # Get dimension names
    if R is not None and hasattr(R, "parameters"):
        param_names = list(R.parameters.keys())
    else:
        param_names = [f"Dim {i}" for i in range(len(moments_all[0]['mean']))]

    n_dims = len(param_names)
    colors = plt.cm.tab10.colors

    # Stack values and errors into arrays of shape [n_rounds, n_dims]
    moment_arrays = {
        k: np.stack([m[k] for m in moments_all], axis=0)
        for k in moment_keys + error_keys
    }

    # Compute differences
    moment_diffs = {
        k: np.abs(np.diff(moment_arrays[k], axis=0))
        for k in moment_keys
    }

    fig, axes = plt.subplots(
        n_moments, 2,
        figsize=(13, 3.0 * n_moments),
        sharex='col',
        gridspec_kw={'hspace': 0, 'wspace': 0.3}  # Tight rows, wider columns
    )

    for i, key in enumerate(moment_keys):
        values = moment_arrays[key]
        errors = moment_arrays[f"{key}_error"]
        diffs = moment_diffs[key]

        for d in range(n_dims):
            color = colors[d % 10]
            axes[i, 0].plot(range(n_rounds), values[:, d], label=param_names[d], color=color)
            axes[i, 0].axvline(end, color='gray', linestyle='--', linewidth=0.8)
            axes[i, 0].fill_between(
                range(n_rounds),
                values[:, d] - errors[:, d],
                values[:, d] + errors[:, d],
                color=color,
                alpha=0.2
            )
            axes[i, 1].semilogy(range(1, n_rounds), diffs[:, d], color=color)

        axes[i, 0].set_ylabel(key.capitalize())
        axes[i, 1].set_ylabel(f"Δ {key}")
        axes[i, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8)

        if i < n_moments - 1:
            axes[i, 0].tick_params(labelbottom=False)
            axes[i, 1].tick_params(labelbottom=False)

    axes[-1, 0].set_xlabel("Training Round")
    axes[-1, 1].set_xlabel("Training Round")

    # Custom manual legend positioning: centered under left column
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    axes[0, 0].set_title("Moment values ± error")
    axes[0, 1].set_title("Differences between rounds")

    plt.subplots_adjust(hspace=0, wspace=0.4, bottom=0.12)  # Give legend room
    plt.show()
