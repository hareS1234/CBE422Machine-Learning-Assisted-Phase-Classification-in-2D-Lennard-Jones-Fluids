# visualize.py
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train_model import PhaseClassifier
import torch
import torch.nn.functional as F
from analysis_tools import compute_rdf

model_file = "balanced_phase_classifier.pth"
dataset_file = "combined_dataset.h5"

model = PhaseClassifier()
model.load_state_dict(torch.load(model_file, map_location="cpu"))
model.eval()

def plot_confusion_matrix(true_labels, pred_labels, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Gas", "Liquid"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def plot_label_distribution(labels, filename):
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar([0,1], counts, color=["blue", "green"])
    plt.xticks([0, 1], ["Gas", "Liquid"])
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(filename)
    plt.close()

def plot_example_predictions(images, true_labels, pred_labels, confidences, filename):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap="viridis")
        ax.set_title(f"True: {int(true_labels[i])}, Pred: {int(pred_labels[i])}\nConf: {confidences[i]:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_and_plot_rdf(positions_2d, filename, box_length=50.0, r_max=25.0, dr=0.5):
    """
    Compute RDF for a given lattice configuration (50x50 array).
    Each cell is treated as a particle coordinate.

    Parameters:
    - positions_2d: A 2D numpy array (50x50) representing the configuration.
    - filename: Output filename for the RDF plot.
    - box_length: The length of the simulation box.
    - r_max: Maximum radius to consider for RDF.
    - dr: Bin width for the RDF calculation.
    """
    # Convert the 50x50 lattice into coordinates
    coords = []
    for x in range(positions_2d.shape[0]):
        for y in range(positions_2d.shape[1]):
            coords.append([x, y])
    coords = np.array(coords, dtype=float)  # shape: (2500, 2)

    # Compute RDF using the analysis_tools function
    # positions_list should be a list of coordinate arrays for multiple configs
    # Here, we have just one configuration
    r, g_r = compute_rdf([coords], box_length, r_max, dr=dr)

    # Plot the RDF
    plt.figure(figsize=(6,4))
    plt.plot(r, g_r, label='g(r)')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Ideal gas level')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_results():
    with h5py.File(dataset_file, "r") as hf:
        positions = hf["positions"][:]  # shape: (num_samples, 50, 50)
        labels = hf["labels"][:]

    # Predict on entire dataset
    preds = []
    confidences = []
    for pos in positions:
        pos_tensor = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)  # shape (1,50,50)
        output = model(pos_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        preds.append(pred.item())
        confidences.append(conf.item())

    preds = np.array(preds)
    confidences = np.array(confidences)

    # Plot confusion matrix
    plot_confusion_matrix(labels, preds, "confusion_matrix.png")
    # Plot label distribution
    plot_label_distribution(labels, "label_distribution.png")
    # Plot example predictions (first 10)
    plot_example_predictions(positions[:10], labels[:10], preds[:10], confidences[:10], "example_prediction.png")

    # Compute and plot RDF for a single sample
    # Let's pick one sample, for example, positions[0]
    compute_and_plot_rdf(positions[0], "rdf_sample_0.png")

if __name__ == "__main__":
    visualize_results()
