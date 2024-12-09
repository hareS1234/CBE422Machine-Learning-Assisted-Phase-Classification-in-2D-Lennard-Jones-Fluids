# combine_datasets.py
import h5py
import numpy as np

def combine_datasets(file1, file2, output_file):
    with h5py.File(file1, "r") as hf1, h5py.File(file2, "r") as hf2:
        positions1 = hf1["positions"][:]
        positions2 = hf2["positions"][:]
        labels1 = np.ones(len(positions1))  # Label liquid as 1
        labels2 = np.zeros(len(positions2)) # Label gas as 0

        combined_positions = np.vstack((positions1, positions2))
        combined_labels = np.hstack((labels1, labels2))

        with h5py.File(output_file, "w") as hf_out:
            hf_out.create_dataset("positions", data=combined_positions)
            hf_out.create_dataset("labels", data=combined_labels)
            print(f"Combined dataset saved to {output_file}")

if __name__ == "__main__":
    combine_datasets("liquid_samples.h5", "gas_samples.h5", "combined_dataset.h5")
