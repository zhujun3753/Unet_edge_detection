from .unet_parts import *


class EncoderDecoder(nn.Module):
    def __init__(self, input_channels):
        super(EncoderDecoder, self).__init__()


        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Add an additional convolutional layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, kernel_size=2, stride=2),
            # nn.Sigmoid() # No need to constrain the values
        )

        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def compute_Image_gradients(x):
    '''
    returns a concatenation of the image, a smooth image and gradients of the image
    '''
    # Convert the image to grayscale
    concatenated_smoothed_batch = []
    for image_ in x:
        bgr_image = cv2.cvtColor(image_.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

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
        gradient_channels = np.zeros((gradient_x.shape[0], gradient_x.shape[1], 3), dtype = np.uint8)
        gradient_channels[:,:,0] = gradient_x_normalized
        gradient_channels[:,:,1] = gradient_y_normalized
        gradient_channels[:,:,2] = gradient_magnitude_normalized

        # Concatenate gradient channels with the original image
        concatenated = np.concatenate((bgr_image, gradient_channels), axis=2)

        smoothed_reshaped = smoothed[:, :, np.newaxis]

        # Concatenate smoothed image with the concatenated image
        concatenated_smoothed = np.concatenate((smoothed_reshaped, concatenated), axis=2)
        concatenated_smoothed_normalized = concatenated_smoothed / 255.0

        concatenated_smoothed_batch.append(concatenated_smoothed_normalized)

     # Convert the list of results to a NumPy array
    concatenated_smoothed_batch = np.array(concatenated_smoothed_batch)
    concatenated_smoothed_batch = np.transpose(concatenated_smoothed_batch, (0, 3, 1, 2))
    return concatenated_smoothed_batch
