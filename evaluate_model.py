import numpy as np
import torch
from unet import  EncoderDecoder, compute_Image_gradients

def validate_avg_precison(dataloader, model, ecn_model, criterion, criterion_reconstruction, device):
    # Put model in evaluation mode
    model.eval()
    ecn_model.eval()
    reconstruction_losses = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            labels = sample_batched['labels'].to(device)

            concatenated_smoothed_batch = compute_Image_gradients(images.cpu().numpy())
            concatenated_smoothed_batch = torch.from_numpy(concatenated_smoothed_batch).to(device)

            # Forward pass through the encoder-decoder network
            encoded, decoded = ecn_model(concatenated_smoothed_batch)
            reconstruction_loss = criterion_reconstruction(decoded, concatenated_smoothed_batch)
            reconstruction_losses.append(reconstruction_loss.item())

            # Forward pass through the edge detection model
            preds_list = model(images, encoded.detach())

            # Calculate precision and recall for each threshold
            for preds in preds_list:
                preds = torch.sigmoid(preds)
                preds_binary = (preds > 0.5).float()  # Assuming a threshold of 0.5
                true_positives = torch.sum(preds_binary * labels).item()
                false_positives = torch.sum(preds_binary * (1 - labels)).item()
                false_negatives = torch.sum((1 - preds_binary) * labels).item()

                precision = true_positives / (true_positives + false_positives + 1e-8)  # Add epsilon to avoid division by zero
                recall = true_positives / (true_positives + false_negatives + 1e-8)  # Add epsilon to avoid division by zero

                precisions.append(precision)
                recalls.append(recall)

    # Calculate average precision and recall
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_precision, avg_recall
