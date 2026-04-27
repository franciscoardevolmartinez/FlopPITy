import json
import os
import glob
import shutil
import warnings

import cloudpickle as pickle
import numpy as np


class RetrievalOutput:
    """Manage retrieval output paths and array archive serialization."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_log_path(self, filename="retrieval_setup.json"):
        return os.path.join(self.output_dir, filename)

    def checkpoint_path(self, filename="retrieval.pkl"):
        return os.path.join(self.output_dir, filename)

    def pre_round_checkpoint_name(self, round_index):
        return f"retrieval_pre_round_{round_index + 1}.pkl"

    def cleanup_pre_round_checkpoints(self, keep_filename=None):
        """Remove old pre-round checkpoints, optionally keeping one file."""
        keep_path = (
            self.checkpoint_path(keep_filename)
            if keep_filename is not None
            else None
        )
        for path in glob.glob(self.checkpoint_path("retrieval_pre_round_*.pkl")):
            if keep_path is not None and os.path.abspath(path) == os.path.abspath(keep_path):
                continue
            os.remove(path)

    def round_dir(self, round_index):
        return os.path.join(self.output_dir, "rounds", f"round_{round_index:03d}")

    def round_data_path(self, round_index):
        return os.path.join(self.round_dir(round_index), "training_data.npz")

    def observations_dir(self):
        return os.path.join(self.output_dir, "observations")

    def posterior_samples_path(self, round_index):
        return os.path.join(
            self.output_dir,
            f"posterior_samples_round_{round_index}.txt",
        )

    def clone_observation_files(self, obs_sources):
        """Copy input observation files into the retrieval output directory."""
        if not obs_sources:
            return {}

        cloned = {}
        os.makedirs(self.observations_dir(), exist_ok=True)
        used_names = set()
        for key, source in obs_sources.items():
            source = os.fspath(source)
            if not os.path.exists(source):
                warnings.warn(
                    f"Observation source {source!r} does not exist and could not "
                    "be copied into the retrieval output.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            filename = self._observation_copy_name(key, source, used_names)
            destination = os.path.join(self.observations_dir(), filename)
            shutil.copy2(source, destination)
            cloned[key] = destination
        return cloned

    def write_posterior_samples(self, round_index, samples, parameter_names):
        """Write posterior samples as a text table."""
        path = self.posterior_samples_path(round_index)
        header = " ".join(str(name) for name in parameter_names)
        np.savetxt(path, np.asarray(samples), header=header)
        return path

    def write_setup_log(self, payload, filename="retrieval_setup.json"):
        path = self.setup_log_path(filename)
        with open(path, "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        return path

    def write_checkpoint(self, retrieval, filename="retrieval.pkl"):
        path = self.checkpoint_path(filename)
        with open(path, "wb") as file:
            pickle.dump(retrieval, file)
        return path

    def write_round_data(
        self,
        round_index,
        thetas,
        spectra,
        nat_thetas=None,
        sample_sources=None,
        processed_spectra=None,
        fitted_radii=None,
    ):
        path = self.round_data_path(round_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        metadata = {
            "format": "floppity.training_data",
            "version": 1,
            "spectra": [],
            "processed_spectra": [],
        }
        arrays = {
            "thetas": np.asarray(thetas),
            "metadata": np.array(json.dumps(metadata)),
        }
        if nat_thetas is not None:
            arrays["nat_thetas"] = np.asarray(nat_thetas)
        if sample_sources is not None:
            arrays["sample_sources"] = np.asarray(sample_sources, dtype="U")
        if fitted_radii is not None:
            arrays["fitted_radii"] = np.asarray(fitted_radii)

        for i, (key, value) in enumerate(spectra.items()):
            array_name = f"spectrum_{i}"
            metadata["spectra"].append(
                {
                    "array": array_name,
                    "key": self._serialize_key(key),
                }
            )
            arrays[array_name] = np.asarray(value)

        if processed_spectra is not None:
            for i, (key, value) in enumerate(processed_spectra.items()):
                array_name = f"processed_spectrum_{i}"
                metadata["processed_spectra"].append(
                    {
                        "array": array_name,
                        "key": self._serialize_key(key),
                    }
                )
                arrays[array_name] = np.asarray(value)

        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_round_data(cls, path):
        path = os.fspath(path)
        if path.endswith(".npz"):
            return cls._load_npz_round_data(path)
        return cls._load_pickle_round_data(path)

    @classmethod
    def _load_npz_round_data(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            spectra = {}
            for entry in metadata["spectra"]:
                key = cls._deserialize_key(entry["key"])
                spectra[key] = archive[entry["array"]]

            result = {
                "par": archive["thetas"],
                "spec": spectra,
            }
            processed_spectra = {}
            for entry in metadata.get("processed_spectra", []):
                key = cls._deserialize_key(entry["key"])
                processed_spectra[key] = archive[entry["array"]]
            if processed_spectra:
                result["post_spec"] = processed_spectra
            if "nat_thetas" in archive:
                result["nat_par"] = archive["nat_thetas"]
            if "sample_sources" in archive:
                result["sample_sources"] = archive["sample_sources"]
            if "fitted_radii" in archive:
                result["fitted_radii"] = archive["fitted_radii"]
            return result

    @staticmethod
    def _load_pickle_round_data(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _serialize_key(key):
        return {
            "type": type(key).__name__,
            "value": str(key),
        }

    @classmethod
    def _observation_copy_name(cls, key, source, used_names):
        stem = cls._safe_filename_part(key)
        basename = os.path.basename(source)
        filename = f"{stem}_{basename}" if stem else basename
        root, ext = os.path.splitext(filename)
        candidate = filename
        index = 2
        while candidate in used_names:
            candidate = f"{root}_{index}{ext}"
            index += 1
        used_names.add(candidate)
        return candidate

    @staticmethod
    def _safe_filename_part(value):
        text = str(value)
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in text
        ).strip("_")

    @staticmethod
    def _deserialize_key(payload):
        key_type = payload["type"]
        value = payload["value"]
        if key_type == "int":
            return int(value)
        if key_type == "float":
            return float(value)
        if key_type == "bool":
            return value == "True"
        return value
