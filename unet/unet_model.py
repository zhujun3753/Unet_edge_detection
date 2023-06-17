""" Full assembly of the parts to form the complete network """
import cv2
import numpy as np
from .unet_parts import *


class EncoderDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EncoderDecoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.encoder_decoder = EncoderDecoder(1024 // factor, 1024 // factor)  # Instantiate the EncoderDecoder module
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        gradients = compute_Image_gradients(x)

        # Pass gradients through the EncoderDecoder model to extract features
        encoded, decoded = self.encoder_decoder(gradients)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
     
        # Concatenate the encoder output with the output of self.down3
        x = self.up1(torch.cat([x5, encoded], dim=1), x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        result_logits = [logits]
        return result_logits, decoded

def compute_Image_gradients(x):
    # Convert the image to grayscale
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute gradients using the Sobel operator
    gradient_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize gradients
    gradient_x_normalized = cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gradient_y_normalized = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create separate channels for gradients
    gradient_channels = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    gradient_channels[:,:,0] = gradient_x_normalized
    gradient_channels[:,:,1] = gradient_y_normalized
    gradient_channels[:,:,2] = gradient_magnitude_normalized

    # Concatenate gradient channels with the original image
    concatenated = np.concatenate((x, gradient_channels), axis=2)

    # Concatenate smoothed image with the concatenated image
    concatenated_smoothed = np.concatenate((smoothed, concatenated), axis=2)

    return concatenated_smoothed
