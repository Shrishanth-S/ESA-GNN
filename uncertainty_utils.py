import torch

def predict_with_uncertainty(model, encoder, decoder, data, T=30):
    model.train()      # 🔥 Dropout ON!
    encoder.train()
    decoder.train()

    # 🔍 Automatically detect the model device (assumes all models on same device)
    device = next(model.parameters()).device

    # 📦 Move input data to that device
    obs = data.obs_seq.to(device)          # [N, obs_len, 2]
    last_pos = obs[:, -1, :].to(device)    # [N, 2]
    edge_index = data.edge_index.to(device)

    preds = []
    for _ in range(T):
        encoded = encoder(obs)                             # [N, 32]
        node_input = torch.cat([last_pos, encoded], dim=1) # [N, 34]
        context = model(node_input, edge_index)            # [N, 64]
        out = decoder(context, last_pos)                   # [N, 12, 2]
        preds.append(out)

    return torch.stack(preds)  # [T, N, 12, 2]
