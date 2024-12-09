import torch
import torch.nn.functional as F
import h5py
from train_model import PhaseClassifier

def predict_config(model, positions):
    positions_tensor = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
    outputs = model(positions_tensor)
    probabilities = F.softmax(outputs, dim=1)
    confidence, pred = torch.max(probabilities, 1)
    return pred.item(), confidence.item()

def infer_and_optimize(model_file, test_file):
    model = PhaseClassifier()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    with h5py.File(test_file, "r") as hf:
        positions = hf["positions"][:]
        labels = hf["labels"][:]

    # Test first 10 samples (likely all liquid)
    print("Testing first 10 samples (likely liquid):")
    correct_liquid = 0
    for i in range(10):
        pred, conf = predict_config(model, positions[i])
        true_label = "Gas" if labels[i] == 0 else "Liquid"
        predicted_label = "Gas" if pred == 0 else "Liquid"
        if pred == labels[i]:
            correct_liquid += 1
        print(f"Sample {i}: True: {true_label}, Predicted: {predicted_label}, Confidence: {conf:.2f}")

    accuracy_liquid = correct_liquid / 10 * 100
    print(f"Accuracy on these 10 (liquid) samples: {accuracy_liquid:.2f}%\n")

    # Test 10 samples from the gas portion, e.g., indices 500–509
    print("Testing samples 500–509 (likely gas):")
    correct_gas = 0
    start_index = 500  # assuming first 500 are liquid, next 500 are gas
    for i in range(start_index, start_index+10):
        pred, conf = predict_config(model, positions[i])
        true_label = "Gas" if labels[i] == 0 else "Liquid"
        predicted_label = "Gas" if pred == 0 else "Liquid"
        if pred == labels[i]:
            correct_gas += 1
        print(f"Sample {i}: True: {true_label}, Predicted: {predicted_label}, Confidence: {conf:.2f}")

    accuracy_gas = correct_gas / 10 * 100
    print(f"Accuracy on these 10 (gas) samples: {accuracy_gas:.2f}%")

if __name__ == "__main__":
    infer_and_optimize("balanced_phase_classifier.pth", "combined_dataset.h5")
