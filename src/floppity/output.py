import json
import os

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

    def round_dir(self, round_index):
        return os.path.join(self.output_dir, "rounds", f"round_{round_index:03d}")

    def round_data_path(self, round_index):
        return os.path.join(self.round_dir(round_index), "training_data.npz")

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
