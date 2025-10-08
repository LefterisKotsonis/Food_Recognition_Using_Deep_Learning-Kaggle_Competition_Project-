import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FoodTestDataset
from models import build_efficientnetv2_l_model, build_efficientnetv2_xl_model, build_convnext_large_model


def load_model(model_fn, weight_path, num_classes, device):
    model = model_fn(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def ensemble_predict(efficientnetv2_l_path, convnext_l_path, efficientnetv2_xl_path, test_image_dir, output_csv, threshold=0.15):
    # Config
    BATCH_SIZE = 64
    NUM_CLASSES = 498
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load test set
    import os
    test_filenames = sorted(os.listdir(test_image_dir))
    test_dataset = FoodTestDataset(test_filenames, test_image_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Load models
    efficientnetv2_l = load_model(build_efficientnetv2_l_model, efficientnetv2_l_path, NUM_CLASSES, DEVICE)
    convnext_l = load_model(build_convnext_large_model, convnext_l_path, NUM_CLASSES, DEVICE)
    efficientnetv2_xl = load_model(build_efficientnetv2_xl_model, efficientnetv2_xl_path, NUM_CLASSES, DEVICE)

    # Predict and average
    all_preds, all_filenames = [], []

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Ensembling"):
            inputs = inputs.to(DEVICE)

            # Get sigmoid probabilities from both models
            probs1 = torch.sigmoid(efficientnetv2_l(inputs))
            probs2 = torch.sigmoid(convnext_l(inputs))
            probs3 = torch.sigmoid(efficientnetv2_xl(inputs))

            # Average predictions
            avg_probs = (probs1 + probs2 + probs3) / 3

            # Apply global threshold
            preds = (avg_probs > threshold).int().cpu().numpy()
            all_preds.extend(preds)
            all_filenames.extend(filenames)

    # Write submission file
    submission = pd.DataFrame(all_preds, columns=[str(i) for i in range(NUM_CLASSES)])
    submission.insert(0, 'Filename', all_filenames)
    submission.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")


# --- Run the following code block in python console to activate ensemble method and produce submission file with the combined models --- #

# from ensemble_predict import ensemble_predict
# ensemble_predict(
#     efficientnetv2_l_path='efficientnetv2_l_food_model.pth',
#     convnext_l_path='convnext_large_food_model.pth',
#     efficientnetv2_xl_path='efficientnetv2_xl_food_model.pth',
#     test_image_dir='images_test',
#     output_csv='submission.csv',
#     threshold=0.25
# )
