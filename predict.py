import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FoodTestDataset
from models import build_convnext_large_model


def generate_submission(model_path, test_image_dir, output_csv, best_thresh=0.5):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test filenames
    test_filenames = sorted(os.listdir(test_image_dir))

    # Define test transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    test_dataset = FoodTestDataset(test_filenames, test_image_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Load model
    model = build_convnext_large_model(num_classes=498)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    all_preds, all_filenames = [], []

    with torch.no_grad():
        for inputs, filenames in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # Threshold-based prediction
            preds = (probs > best_thresh).int().cpu().numpy()

            all_preds.extend(preds)
            all_filenames.extend(filenames)

    # Create submission DataFrame
    submission = pd.DataFrame(all_preds, columns=[str(i) for i in range(498)])
    submission.insert(0, 'Filename', all_filenames)

    # Save to CSV
    submission.to_csv(output_csv, index=False)
    print(f"Submission saved to: {output_csv}")