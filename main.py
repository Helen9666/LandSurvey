import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import get_model
from dataset import get_dataloaders
from engine import train_one_epoch, evaluate
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train object detection models for drone imagery')
    parser.add_argument('--data_path', required=True, help='Path to dataset')
    parser.add_argument('--model', default='faster_rcnn', choices=['faster_rcnn', 'retinanet', 'yolo'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes including background')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size)
    
    # Get model
    model = get_model(args.model, args.num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        evaluate(model, val_loader, device)
        
        # Save checkpoint
        utils.save_checkpoint(
            model, optimizer, epoch, 0,
            os.path.join(args.output_dir, f"{args.model}_epoch_{epoch}.pth")
        )
    
    # Save final model
    torch.save(model, os.path.join(args.output_dir, f"{args.model}_final.pth"))
    print(f"Training complete. Model saved to {args.output_dir}/{args.model}_final.pth")

if __name__ == "__main__":
    main()