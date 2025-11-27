import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_configs import CONFIG
from model import SymphonyClassifier

def inference(folder_path, model_name, mode):
    test_data = np.load(os.path.join(folder_path, "test.npz"))
    X_test = test_data["X"]
    y_composer_test = test_data["y_composer"]
    y_era_test = test_data["y_era"]
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_composer_test).long()
    y_era_test_tensor = torch.from_numpy(y_era_test).long()
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor, y_era_test_tensor)
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SymphonyClassifier(
        input_size=X_test.shape[2],
        n_embedding=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)

    model_path = os.path.join(model_name + ".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    total_composer = 0
    total_era = 0
    correct_composer = 0
    correct_era = 0

    with torch.no_grad():
        for X_test, composer_labels, era_labels in test_loader:
            X_test = X_test.to(device)
            composer_labels = composer_labels.to(device)
            era_labels = era_labels.to(device)

            if mode == "composer_era":
                composer_output, era_output = model.forward_composer_era(X_test, device=device)
            elif mode == "composer":
                composer_output = model.forward_composer(X_test, device=device)
                era_output = None
            elif mode == "era":
                composer_output = None
                era_output = model.forward_era(X_test, device=device)
            
            # Compute accuracy
            if composer_output is not None:
                _, predicted_composer = torch.max(composer_output, 1)
                correct_composer += (predicted_composer == composer_labels).sum().item()
                total_composer += composer_labels.size(0)

            if era_output is not None:
                _, predicted_era = torch.max(era_output, 1)
                correct_era += (predicted_era == era_labels).sum().item()
                total_era += era_labels.size(0)

        # Compute percentages
        test_acc_composer = 100 * correct_composer / total_composer if total_composer > 0 else None
        test_acc_era = 100 * correct_era / total_era if total_era > 0 else None

        print(f'Test Composer Acc: {test_acc_composer if test_acc_composer is not None else "N/A"}, '
              f'Test Era Acc: {test_acc_era if test_acc_era is not None else "N/A"}')

if __name__ == "__main__":
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the data path, model_name, and mode as argument, e.g., python inference.py ML/dataset best_model composer_era.')
    data_folder = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    inference(data_folder, model_name, mode)