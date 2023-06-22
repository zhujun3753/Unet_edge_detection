""" Full assembly of the parts to form the complete network """
from .encoder_decoder import *
from .unet_parts import *



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
        self.encoder_decoder = EncoderDecoder(input_channels = 7)  # Instantiate the EncoderDecoder module
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.catc = DoubleConv(1024, 512)

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        gradients_np = compute_Image_gradients(x_np)
        # Pass gradients through the EncoderDecoder model to extract features
        # Convert the gradients back to a PyTorch tensor
        gradients = torch.from_numpy(gradients_np).to(x.device)
        encoded, decoded = self.encoder_decoder(gradients)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
     
        # Concatenate the encoder output with the output of self.down3
        # import pdb;pdb.set_trace()
        x5_encoded = self.catc(torch.cat([x5, encoded], dim=1))
        x = self.up1(x5_encoded, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        result_logits = [logits]
        return result_logits, decoded


# encoded shape
#torch.Size([16, 256, 44, 44])

#x5 [16, 512, 22 ,22]

#x4 [16, 512, 44 ,22]