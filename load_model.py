import torch
from model import GAT, EncoderLSTM, DecoderLSTM
from dataset import PedestrianDataset
from visualize_uncertainty import visualize_uncertainty
from torch.utils.data import Subset

# Load checkpoint (assuming it contains state_dicts and metadata)
checkpoint = torch.load("saved_models/final_model_with_loss.pt")

# Re-create model architectures
encoder = EncoderLSTM()
gat = GAT(in_channels=34)
decoder = DecoderLSTM(pred_len=12)

# Load the weights
encoder.load_state_dict(checkpoint["encoder_state_dict"])
gat.load_state_dict(checkpoint["gat_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

print(f"✅ Loaded model from epoch {checkpoint.get('epoch', '?')} with total loss: {checkpoint['total_loss']:.4f}")


# Load dataset again (same way you did during training)
dataset = Subset(PedestrianDataset("data/annotations/zara01/world_coordinate_inter.csv"), range(200))

# Visualize prediction + uncertainty
visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=60, T=100)