import segmentation_models_pytorch as smp

def get_model(encoder, encoder_weights, activation='sigmoid', num_classes=1, in_channels=3):
    return smp.Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes=num_classes, 
        in_channels=in_channels
    )
