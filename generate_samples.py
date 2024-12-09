import numpy as np
import h5py
import sys

def generate_samples(num_samples, grid_size, save_path, phase):
    data = []
    for _ in range(num_samples):
        if phase == "liquid":
            # Liquid: Values concentrated in the high range [0.7, 1.0]
            positions = np.random.uniform(0.7, 1.0, size=(grid_size, grid_size))
        elif phase == "gas":
            # Gas: Values concentrated in the low range [0.0, 0.3]
            positions = np.random.uniform(0.0, 0.3, size=(grid_size, grid_size))
        else:
            raise ValueError("phase must be 'liquid' or 'gas'")
        data.append(positions)

    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("positions", data=np.array(data))
        print(f"Generated {num_samples} {phase} samples with shape ({grid_size}, {grid_size}) saved to {save_path}")

if __name__ == "__main__":
    # Generate 500 liquid and 500 gas samples
    generate_samples(500, 50, "liquid_samples.h5", "liquid")
    generate_samples(500, 50, "gas_samples.h5", "gas")
