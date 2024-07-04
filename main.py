import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging
from model import get_model
from dataset import SatelliteDataset, get_train_transform, get_val_transform, get_test_transform
from utils import seed_everything, seed_worker, DiceBCELoss, train_epoch, eval_epoch, infer
from parser import parse_args

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(42)

    # Logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data = pd.read_csv(args.train_csv)
    valid_data = pd.read_csv(args.valid_csv)
    test_data = pd.read_csv(args.test_csv)

    # Data augmentation and normalization for training
    train_transform = get_train_transform(args.img_size)
    val_transform = get_val_transform(args.img_size)
    test_transform = get_test_transform(args.img_size)

    # Create datasets and dataloaders
    train_dataset = SatelliteDataset(train_data, transform=train_transform)
    valid_dataset = SatelliteDataset(valid_data, transform=val_transform)
    test_dataset = SatelliteDataset(test_data, transform=test_transform, infer=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=seed_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, optimizer, and loss function
    model = get_model(args.encoder, args.encoder_weights)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = DiceBCELoss()

    # Training and evaluation
    best_val_dice = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = eval_epoch(model, valid_loader, loss_fn, device)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val dice score: {val_dice:.4f}")
        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val dice score: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.model_save_path)
            logging.info(f"Epoch: {epoch}, best DICE score: {best_val_dice:.4f}")
        print(f"best DICE score: {best_val_dice:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(args.model_save_path))

    # Inference
    test_predictions = infer(model, test_loader, device)

    submit = pd.read_csv(args.sample_submission_csv)
    submit['mask_rle'] = test_predictions
    submit.to_csv(args.model_save_path.replace('.pt', '.csv'), index=False)

if __name__ == "__main__":
    main()
