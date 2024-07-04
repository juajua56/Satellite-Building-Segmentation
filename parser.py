import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a UNet model for satellite image segmentation')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--encoder', type=str, default='tu-tf_efficientnetv2_m', help='Encoder model for UNet')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='Pre-trained weights for encoder')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for training and validation')
    
    # Data paths
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help='Path to the train CSV file')
    parser.add_argument('--valid_csv', type=str, default='/data/valid.csv', help='Path to the validation CSV file')
    parser.add_argument('--test_csv', type=str, default='/data/test.csv', help='Path to the test CSV file')
    parser.add_argument('--sample_submission_csv', type=str, default='./result/sample_submission.csv', help='Path to the sample submission CSV file')
    
    # Other parameters
    parser.add_argument('--log_file', type=str, default='result.log', help='Log file name')
    parser.add_argument('--model_save_path', type=str, default='./result.pt', help='Path to save the best model')

    return parser.parse_args()
