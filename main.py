import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pandas import read_csv

from dataset import FoodTrainDataset
from models import build_convnext_large_model
from train import train_model
from predict import generate_submission

# Configuration
CSV_PATH = 'train_onehot.csv'
IMAGE_DIR = 'images_train'
TEST_IMAGE_DIR = 'images_test'
BATCH_SIZE = 16

EPOCHS = 20
NUM_CLASSES = 498
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def main():
    # Load and split dataset
    df = read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = FoodTrainDataset(train_df, IMAGE_DIR, transform=train_transform)
    val_dataset = FoodTrainDataset(val_df, IMAGE_DIR, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Build model
    model = build_convnext_large_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Optimizer and LR scheduler
    optimizer = (torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train model
    model, best_thresh = train_model(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=EPOCHS, patience=4)

    # Save best model
    torch.save(model.state_dict(), 'convnext_large_food_model.pth')
    print("Model saved to convnext_large_food_model.pth")

    # Generate submission from test images
    generate_submission(
        model_path='convnext_large_food_model.pth',
        test_image_dir=TEST_IMAGE_DIR,
        output_csv='submission.csv',
        best_thresh=best_thresh
    )


# Entry point for multiprocessing safety (Windows)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
