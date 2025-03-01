import argparse
# Import necessary modules and functions
from model import build_model, train_model  # Example imports
from utils import load_data

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg13", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_layer_1_units", type=int, default=512, help="Number of neurons/units in first hidden layer")
    parser.add_argument("--hidden_layer_2_units", type=int, default=256, help="Number of neurons/units in second hidden layer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Load the data
    trainloader, validloader, testloader = load_data(args.data_directory)
    # Build the model
    model = build_model(args.arch, args.hidden_layer_1_units, args.hidden_layer_2_units)
    # Train the model
    train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)
    # Save the checkpoint
    model.save_checkpoint(args.save_dir)
    
if __name__ == '__main__':
    main()
