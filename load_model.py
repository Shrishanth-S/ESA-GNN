import torch
from model import GAT, EncoderLSTM, DecoderLSTM
from dataset import PedestrianDataset
from visualize_prediction import predict_and_visualize
from visualize_uncertainty import visualize_uncertainty
from torch.utils.data import Subset

# Load checkpoint (assuming it contains state_dicts and metadata)
checkpoint = torch.load("saved_models/model_seq_eth_epoch75_ade0.2213_fde0.3860.pt")

# Re-create model architectures
encoder = EncoderLSTM()
gat = GAT(in_channels=34)
decoder = DecoderLSTM(pred_len=12)

# Load the weights
encoder.load_state_dict(checkpoint["encoder_state_dict"])
gat.load_state_dict(checkpoint["gat_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

print(f"✅ Loaded model")


# Load dataset again (same way you did during training)
dataset = PedestrianDataset("data/annotations/seq_eth/world_coordinate_inter.csv")

predict_and_visualize(gat, encoder, decoder, dataset, sample_index=679)

# Visualize prediction + uncertainty
visualize_uncertainty(gat, encoder, decoder, dataset, sample_index=679, T=100)