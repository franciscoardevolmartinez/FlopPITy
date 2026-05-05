#!/usr/bin/env python
"""Save the retrieval diagnostic plots from the notebook to image files.

Examples
--------
python examples/plot_retrieval_results.py /path/to/output_FlopPITy
python examples/plot_retrieval_results.py /path/to/output_FlopPITy_ensemble --round latest --format png pdf
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path


RADIUS_LABEL = "fitted_radius"


def load_runtime_dependencies():
    """Import plotting/science dependencies after argparse handles --help."""
    global plt, withStroke, np, corner, Retrieval, RetrievalOutput

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke
    import numpy as np
    from corner import corner

    from floppity import Retrieval, RetrievalOutput


def is_ensemble_output(output_dir):
    return (output_dir / "aggregated").exists() and any(output_dir.glob("member_*"))


def resolve_output_paths(output_dir, use_ensemble="auto", retrieval_path=None):
    output_dir = Path(output_dir)
    if use_ensemble == "auto":
        use_ensemble = is_ensemble_output(output_dir)

    if use_ensemble:
        data_dir = output_dir / "aggregated"
        member_dirs = sorted(path for path in output_dir.glob("member_*") if path.is_dir())
        if not member_dirs:
            raise FileNotFoundError(f"No member_* directories found in ensemble output {output_dir}")
        retrieval_path = Path(retrieval_path) if retrieval_path is not None else member_dirs[0] / "retrieval.pkl"
        return {
            "root_dir": output_dir,
            "data_dir": data_dir,
            "retrieval_path": retrieval_path,
            "is_ensemble": True,
            "member_dirs": member_dirs,
        }

    retrieval_path = Path(retrieval_path) if retrieval_path is not None else output_dir / "retrieval.pkl"
    return {
        "root_dir": output_dir,
        "data_dir": output_dir,
        "retrieval_path": retrieval_path,
        "is_ensemble": False,
        "member_dirs": [],
    }


def available_rounds(data_dir):
    round_root = data_dir / "rounds"
    rounds = []
    for path in sorted(round_root.glob("round_*")):
        try:
            rounds.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    return rounds


def resolve_round(data_dir, selected_round):
    rounds = available_rounds(data_dir)
    if not rounds:
        raise FileNotFoundError(f"No round directories found in {data_dir / 'rounds'}")
    if selected_round == "latest":
        return rounds[-1]
    selected_round = int(selected_round)
    if selected_round not in rounds:
        raise FileNotFoundError(f"Round {selected_round} is not available. Available rounds: {rounds}")
    return selected_round


def round_data_path(data_dir, round_index):
    return data_dir / "rounds" / f"round_{round_index:03d}" / "training_data.npz"


def posterior_samples_path(data_dir, round_index):
    return data_dir / f"posterior_samples_round_{round_index + 1}.txt"


def atmosphere_path(data_dir, round_index):
    candidates = [
        data_dir / "arcis_files" / f"mixingratios_round_{round_index}.dat",
        data_dir / "arcis_outputs" / "arcis_files" / f"mixingratios_round_{round_index}.dat",
        data_dir / f"mixingratios_round_{round_index}.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No atmosphere file found for round "
        f"{round_index}. Checked: " + ", ".join(str(path) for path in candidates)
    )


def proposal_for_corner(retrieval, alpha):
    if alpha > 0:
        if getattr(retrieval, "posteriors", None):
            return retrieval.posteriors[-1]
        proposal = retrieval.proposals[-1]
        if hasattr(proposal, "posterior"):
            return proposal.posterior
    return retrieval.proposals[-1]


def proposal_sample_mask(round_data, alpha, round_index):
    sources = round_data.get("sample_sources")
    if alpha <= 0 or round_index == 0:
        return slice(None)
    if sources is None:
        raise ValueError("This run used alpha > 0, but the round archive has no sample_sources.")
    return sources == "proposal"


def select_rows(values, max_rows=None, rng_seed=12345):
    values = np.asarray(values)
    if max_rows is None or len(values) <= max_rows:
        return values
    rng = np.random.default_rng(rng_seed)
    indices = np.sort(rng.choice(len(values), size=max_rows, replace=False))
    return values[indices]


def round_corner_samples(retrieval, round_data, mask, max_samples=None):
    parameter_samples = np.asarray(round_data.get("nat_par", round_data["par"]))[mask]
    labels = list(retrieval.parameters.keys())
    if "fitted_radii" in round_data:
        fitted_radii = np.asarray(round_data["fitted_radii"])[mask].reshape(-1, 1)
        parameter_samples = np.column_stack([parameter_samples, fitted_radii])
        labels = labels + [RADIUS_LABEL]
    return select_rows(parameter_samples, max_rows=max_samples), labels


def posterior_corner_samples(retrieval, alpha, max_samples):
    proposal = proposal_for_corner(retrieval, alpha)
    samples = proposal.sample((max_samples,)).detach().cpu().numpy()
    samples = samples.reshape(-1, len(retrieval.parameters))
    return samples, list(retrieval.parameters.keys())


def posterior_file_corner_samples(retrieval, data_dir, round_index, max_samples):
    path = posterior_samples_path(data_dir, round_index)
    if not path.exists():
        return None
    samples = np.asarray(np.loadtxt(path))
    if samples.ndim == 1:
        if len(retrieval.parameters) == 1:
            samples = samples.reshape(-1, 1)
        else:
            samples = samples.reshape(1, -1)
    return select_rows(samples, max_rows=max_samples), list(retrieval.parameters.keys())


def corner_samples(retrieval, round_data, mask, alpha, max_samples, data_dir, round_index, is_ensemble):
    if "fitted_radii" in round_data:
        return round_corner_samples(retrieval, round_data, mask, max_samples=max_samples)
    if is_ensemble:
        from_file = posterior_file_corner_samples(retrieval, data_dir, round_index, max_samples)
        if from_file is not None:
            return from_file
        return round_corner_samples(retrieval, round_data, mask, max_samples=max_samples)
    return posterior_corner_samples(retrieval, alpha, max_samples)


def sigma_bands(values):
    return {
        "3sigma": np.nanpercentile(values, [0.135, 99.865], axis=0),
        "2sigma": np.nanpercentile(values, [2.275, 97.725], axis=0),
        "1sigma": np.nanpercentile(values, [15.865, 84.135], axis=0),
    }


def best_fit_index(retrieval, spectra):
    obs_items = list(retrieval.obs.items())
    n_models = spectra[obs_items[0][0]].shape[0]
    chi2 = np.zeros(n_models)
    dof = 0
    for obs_key, obs in obs_items:
        spec = spectra[obs_key]
        dof += len(spec[0])
        chi2 += np.nansum(((spec - obs[:, 1]) / obs[:, 2]) ** 2, axis=1)
    best_idx = int(np.nanargmin(chi2))
    return best_idx, chi2, dof


def parse_metadata(line):
    metadata = {}
    for item in line.lstrip("#").split():
        if "=" in item:
            key, value = item.split("=", 1)
            metadata[key] = value
    return metadata


def parse_columns(line):
    return line.split("=", 1)[1].split()


def parse_mixingratios(path):
    blocks = []
    current = []
    pending_metadata = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("# ensemble_member="):
            pending_metadata = parse_metadata(line)
            continue
        if line.startswith("# columns=") and current:
            blocks.append((pending_metadata, current))
            current = []
        current.append(line)
    if current:
        blocks.append((pending_metadata, current))

    profiles = []
    for block_metadata, block in blocks:
        columns = None
        metadata = dict(block_metadata)
        numeric = []
        for line in block:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# columns="):
                columns = parse_columns(stripped)
                continue
            if stripped.startswith("# round="):
                metadata.update(parse_metadata(stripped))
                continue
            if stripped.startswith("#"):
                continue
            numeric.append(stripped)
        if not numeric:
            continue
        arr = np.loadtxt(StringIO("\n".join(numeric)))
        arr = np.atleast_2d(arr)
        if columns is None:
            raise ValueError(
                f"No '# columns=' metadata found for an atmosphere block in {path}. "
                "Re-run ARCiS with the current FlopPITy branch so the header is preserved."
            )
        if len(columns) != arr.shape[1]:
            raise ValueError(f"Column count mismatch in {path}: {len(columns)} names for {arr.shape[1]} columns")
        profiles.append({
            "columns": columns,
            "metadata": metadata,
            "data": {name: arr[:, i] for i, name in enumerate(columns)},
        })
    return profiles


def filter_profiles_by_source(profiles, round_data, alpha, round_index):
    sources = round_data.get("sample_sources")
    if alpha <= 0 or round_index == 0:
        return profiles
    if sources is None:
        raise ValueError("This run used alpha > 0, but training_data.npz has no sample_sources.")
    selected = []
    samples_per_member = None
    if profiles:
        members = sorted({int(p["metadata"].get("ensemble_member", 1)) for p in profiles})
        if len(members) > 1:
            samples_per_member = len(sources) // len(members)
    for profile in profiles:
        global_model = int(profile["metadata"].get("global_model", -1))
        member = int(profile["metadata"].get("ensemble_member", 1))
        source_index = global_model
        if samples_per_member is not None:
            source_index = (member - 1) * samples_per_member + global_model
        if 0 <= source_index < len(sources) and sources[source_index] == "proposal":
            selected.append(profile)
    return selected


def profile_grid(profiles, value_name, log_value=False, n_grid=200):
    ranges = [
        (np.nanmin(p["data"]["P"]), np.nanmax(p["data"]["P"]))
        for p in profiles
        if value_name in p["data"]
    ]
    if not ranges:
        raise KeyError(f"No profiles contain column {value_name!r}")
    p_min = max(low for low, high in ranges)
    p_max = min(high for low, high in ranges)
    pressure = np.geomspace(p_min, p_max, n_grid)
    log_pressure = np.log10(pressure)
    values = []
    for profile in profiles:
        data = profile["data"]
        if value_name not in data:
            continue
        p = np.asarray(data["P"])
        v = np.asarray(data[value_name])
        order = np.argsort(p)
        y = np.log10(np.clip(v[order], 1e-300, None)) if log_value else v[order]
        values.append(np.interp(log_pressure, np.log10(p[order]), y))
    return pressure, np.asarray(values)


def component_label(profile):
    thread = profile["metadata"].get("thread", "")
    marker = "_component"
    if marker not in thread:
        return "component 1"
    suffix = thread.split(marker, 1)[1]
    digits = "".join(char for char in suffix if char.isdigit())
    return f"component {digits or suffix}"


def component_sort_key(label):
    digits = "".join(char for char in label if char.isdigit())
    return int(digits) if digits else label


def profiles_by_component(profiles):
    groups = {}
    for profile in profiles:
        groups.setdefault(component_label(profile), []).append(profile)
    return dict(sorted(groups.items(), key=lambda item: component_sort_key(item[0])))


def component_subplots(n_components, width=5, height=6, layout="horizontal"):
    if layout == "vertical":
        fig, axes = plt.subplots(n_components, 1, figsize=(width, height * n_components), squeeze=False, sharey=True)
    else:
        fig, axes = plt.subplots(1, n_components, figsize=(width * n_components, height), squeeze=False, sharey=True)
    return fig, axes.ravel()


def orient_pressure_axis(axes, pressure):
    pressure = np.asarray(pressure)
    for ax in np.ravel(axes):
        ax.set_yscale("log")
        ax.set_ylim(np.nanmax(pressure), np.nanmin(pressure))


def save_figure(fig, figures_dir, stem, formats, dpi):
    paths = []
    for fmt in formats:
        path = figures_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_corner_figure(context, args):
    truth = np.asarray(context["round_data"].get("nat_par", context["round_data"]["par"]))[context["best_idx"]]
    if "fitted_radii" in context["round_data"] and len(context["corner_labels"]) == len(truth) + 1:
        truth = np.append(truth, context["round_data"]["fitted_radii"][context["best_idx"]])
    if len(truth) != len(context["corner_labels"]):
        truth = None

    fig = corner(
        context["corner_samples"],
        labels=context["corner_labels"],
        smooth=args.corner_smooth,
        truths=truth,
        show_titles=True,
    )
    if "fitted_radii" in context["round_data"]:
        title = f"Round {context['round_index']} samples with fitted radius"
    elif context["paths"]["is_ensemble"]:
        title = f"Ensemble posterior samples, round {context['round_index']}"
    elif context["alpha"] > 0:
        title = "Uninflated posterior"
    else:
        title = "Posterior"
    fig.suptitle(title, y=1.02)
    return fig


def plot_retrieved_spectra(context, split_wvl):
    obs_items = list(context["retrieval"].obs.items())
    spectra = context["spectra"]
    mask = context["mask"]
    fig, axes = plt.subplots(
        len(obs_items),
        2,
        figsize=(16, 5 * len(obs_items)),
        gridspec_kw={"width_ratios": [1, 2], "wspace": 0.005},
        sharey=False,
        squeeze=False,
    )

    for i, (obs_key, obs) in enumerate(obs_items):
        spec = spectra[obs_key][mask]
        median = np.nanmedian(spec, axis=0)
        bands = sigma_bands(spec)
        wvl = obs[:, 0]
        flux = obs[:, 1]
        err = obs[:, 2]
        short = wvl < split_wvl
        long = wvl >= split_wvl

        for ax, m in zip(axes[i], [short, long]):
            if not np.any(m):
                continue
            for name, alpha_fill in [("3sigma", 0.10), ("2sigma", 0.16), ("1sigma", 0.26)]:
                lo, hi = bands[name]
                ax.fill_between(wvl[m], 1e3 * lo[m], 1e3 * hi[m], color="tab:blue", alpha=alpha_fill, lw=0)
            ax.plot(wvl[m], 1e3 * median[m], color="tab:blue", lw=2)
            ax.errorbar(wvl[m], 1e3 * flux[m], yerr=1e3 * err[m], fmt="-", ms=3, color="black")
            ax.set_xlim(wvl[m].min(), wvl[m].max())

        axes[i, 0].set_ylabel("Flux (mJy)")
        axes[i, 1].yaxis.set_label_position("right")
        axes[i, 1].yaxis.tick_right()
        axes[i, 0].spines["right"].set_visible(True)
        axes[i, 1].spines["left"].set_visible(True)
        axes[i, 0].tick_params(right=False)
        axes[i, 1].tick_params(left=False)

    axes[-1, 0].set_xlabel("Wavelength (micron)")
    axes[-1, 1].set_xlabel("Wavelength (micron)")
    axes[0, 0].set_ylim(-1e-3, 0.03)
    return fig


def plot_best_fit_spectrum(context, split_wvl):
    obs_items = list(context["retrieval"].obs.items())
    spectra = context["spectra"]
    best_idx = context["best_idx"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
        gridspec_kw={"width_ratios": [1, 2], "wspace": 0.01},
        sharey=False,
    )

    for obs_key, obs in obs_items:
        best_spec = spectra[obs_key][best_idx]
        wvl = obs[:, 0]
        flux = obs[:, 1]
        err = obs[:, 2]
        short = wvl < split_wvl
        long = wvl >= split_wvl

        for ax, m in zip(axes, [short, long]):
            if not np.any(m):
                continue
            ax.errorbar(
                wvl[m],
                1e3 * flux[m],
                yerr=1e3 * err[m],
                fmt="-",
                ms=3,
                lw=3,
                color="black",
                label="data" if obs_key == obs_items[0][0] else None,
            )
            ax.plot(
                wvl[m],
                1e3 * best_spec[m],
                color="salmon",
                lw=2,
                label="best fit" if obs_key == obs_items[0][0] else None,
                zorder=10,
                path_effects=[withStroke(linewidth=4, foreground="white")],
            )

    short_wvls = [obs[:, 0][obs[:, 0] < split_wvl] for _, obs in obs_items if np.any(obs[:, 0] < split_wvl)]
    long_wvls = [obs[:, 0][obs[:, 0] >= split_wvl] for _, obs in obs_items if np.any(obs[:, 0] >= split_wvl)]
    if short_wvls:
        short_wvls = np.concatenate(short_wvls)
        axes[0].set_xlim(short_wvls.min(), short_wvls.max())
    if long_wvls:
        long_wvls = np.concatenate(long_wvls)
        axes[1].set_xlim(long_wvls.min(), long_wvls.max())

    axes[0].set_ylabel("Flux (mJy)")
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel("Flux (mJy)")
    axes[0].spines["right"].set_visible(True)
    axes[1].spines["left"].set_visible(True)
    axes[0].tick_params(right=False)
    axes[1].tick_params(left=False)
    fig.supxlabel("Wavelength (micron)")
    axes[0].legend()
    plt.subplots_adjust(wspace=0)
    return fig


def plot_temperature(component_profiles):
    fig, ax = plt.subplots(figsize=(6.5, 7))
    component_colors = plt.cm.tab10(np.linspace(0, 1, max(len(component_profiles), 1)))
    last_pressure = None
    for color, (component_name, component_group) in zip(component_colors, component_profiles.items()):
        pressure, temperature = profile_grid(component_group, "T", log_value=False)
        last_pressure = pressure
        bands = sigma_bands(temperature)
        for name, alpha_fill in [("3sigma", 0.08), ("2sigma", 0.13), ("1sigma", 0.22)]:
            lo, hi = bands[name]
            ax.fill_betweenx(pressure, lo, hi, color=color, alpha=alpha_fill, lw=0)
        ax.plot(np.nanmedian(temperature, axis=0), pressure, color=color, lw=2, label=f"{component_name} median T")
    orient_pressure_axis([ax], last_pressure)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_abundances(component_profiles, species):
    fig, axes = component_subplots(len(component_profiles), width=16, height=8, layout="vertical")
    last_pressure = None
    legend_handles = []
    legend_labels = []
    for ax, (component_name, component_group) in zip(axes, component_profiles.items()):
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(species), 1)))
        for color, species_name in zip(colors, species):
            pressure, abundance = profile_grid(component_group, species_name, log_value=True)
            last_pressure = pressure
            lo, hi = sigma_bands(abundance)["1sigma"]
            median = np.nanmedian(abundance, axis=0)
            ax.fill_betweenx(pressure, lo, hi, color=color, alpha=0.16, lw=0)
            line, = ax.plot(median, pressure, color=color, lw=2, label=species_name)
            if species_name not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(species_name)
        ax.set_title(component_name)
        ax.set_xlabel("log10 abundance")
    orient_pressure_axis(axes, last_pressure)
    for ax in axes:
        ax.set_ylabel("Pressure [bar]")
        ax.set_xlim(-10, -2)
    fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(0.8, 0.5), frameon=False)
    fig.subplots_adjust(hspace=0.0, right=0.78)
    return fig


def load_context(args):
    paths = resolve_output_paths(args.output_dir, use_ensemble=args.use_ensemble, retrieval_path=args.retrieval_path)
    data_dir = paths["data_dir"]
    retrieval = Retrieval.load(paths["retrieval_path"])
    alpha = float(getattr(retrieval, "alpha", 0))
    round_index = resolve_round(data_dir, args.round)
    data_path = round_data_path(data_dir, round_index)
    round_data = RetrievalOutput.load_round_data(data_path)
    spectra = round_data.get("post_spec", round_data["spec"])
    mask = proposal_sample_mask(round_data, alpha, round_index)
    best_idx, chi2, dof = best_fit_index(retrieval, spectra)
    corner_data, corner_labels = corner_samples(
        retrieval,
        round_data,
        mask,
        alpha,
        max_samples=args.n_corner_samples,
        data_dir=data_dir,
        round_index=round_index,
        is_ensemble=paths["is_ensemble"],
    )
    return {
        "paths": paths,
        "data_dir": data_dir,
        "data_path": data_path,
        "retrieval": retrieval,
        "alpha": alpha,
        "round_index": round_index,
        "round_data": round_data,
        "spectra": spectra,
        "mask": mask,
        "best_idx": best_idx,
        "chi2": chi2,
        "dof": dof,
        "corner_samples": corner_data,
        "corner_labels": corner_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Retrieval output directory or ensemble root.")
    parser.add_argument("--round", default="latest", help="Round index to plot, or 'latest'.")
    parser.add_argument("--use-ensemble", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--retrieval-path", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None, help="Defaults to OUTPUT_DIR/Figures.")
    parser.add_argument("--format", dest="formats", nargs="+", default=["png"], help="Image formats, e.g. png pdf.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--n-corner-samples", type=int, default=1000)
    parser.add_argument("--corner-smooth", type=float, default=0.7)
    parser.add_argument("--spectrum-split-wvl", type=float, default=3.4)
    parser.add_argument("--best-fit-split-wvl", type=float, default=3.2)
    parser.add_argument("--skip-atmosphere", action="store_true", help="Skip temperature/abundance plots.")
    return parser.parse_args()


def normalized_use_ensemble(value):
    if value == "auto":
        return value
    return value == "true"


def main():
    args = parse_args()
    load_runtime_dependencies()
    args.use_ensemble = normalized_use_ensemble(args.use_ensemble)
    figures_dir = args.figures_dir or (args.output_dir / "Figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(args)
    print(f"Loaded retrieval: {context['paths']['retrieval_path']}")
    print(f"Mode: {'ensemble' if context['paths']['is_ensemble'] else 'single retrieval'}")
    print(f"Data directory: {context['data_dir']}")
    print(f"Selected round: {context['round_index']}")
    print(f"Loaded round data: {context['data_path']}")
    print(f"alpha = {context['alpha']}")
    if context["paths"]["is_ensemble"]:
        print(f"Members: {len(context['paths']['member_dirs'])}")
    if isinstance(context["mask"], np.ndarray):
        print(f"Using {context['mask'].sum()} proposal-sampled samples out of {len(context['mask'])} total samples")
    print(f"Best reduced chi2: {np.nanmin(context['chi2'] / context['dof']):.6g}")
    print(f"Writing figures to: {figures_dir}")

    saved = []
    saved += save_figure(plot_corner_figure(context, args), figures_dir, "corner", args.formats, args.dpi)
    saved += save_figure(plot_retrieved_spectra(context, args.spectrum_split_wvl), figures_dir, "retrieved_spectra", args.formats, args.dpi)
    saved += save_figure(plot_best_fit_spectrum(context, args.best_fit_split_wvl), figures_dir, "best_fit_spectrum", args.formats, args.dpi)

    if not args.skip_atmosphere:
        try:
            atm_path = atmosphere_path(context["data_dir"], context["round_index"])
            profiles_all = parse_mixingratios(atm_path)
            profiles = filter_profiles_by_source(profiles_all, context["round_data"], context["alpha"], context["round_index"])
            component_profiles = profiles_by_component(profiles)
            columns = profiles[0]["columns"]
            species = [name for name in columns if name not in {"T", "P", "H2", "He"}]
            saved += save_figure(plot_temperature(component_profiles), figures_dir, "temperature_structure", args.formats, args.dpi)
            saved += save_figure(plot_abundances(component_profiles, species), figures_dir, "molecular_abundances", args.formats, args.dpi)
        except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
            print(f"Skipping atmosphere plots: {exc}")

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
