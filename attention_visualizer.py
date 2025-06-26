import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import world_to_gps, gps_to_pixel
from model import GAT, EncoderLSTM

@torch.no_grad()
def visualize_attention_on_video(
    gat,
    encoder,
    dataset,
    sample_index,
    video_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat.to(device).eval()
    encoder.to(device).eval()

    # === Load sample ===
    data = dataset[sample_index]
    obs = data.obs_seq.to(device)         # [N, obs_len, 2]
    last_pos = obs[:, -1, :]              # [N, 2]
    edge_index = data.edge_index.to(device)

    # === Get the frame number (added during preprocessing)
    if hasattr(data, "frame_id"):
        frame_index = data.frame_id
    else:
        raise AttributeError("❌ Dataset sample missing 'frame_id'. Fix dataset caching.")

    # === Forward pass ===
    encoded = encoder(obs)
    node_input = torch.cat([last_pos, encoded], dim=1)
    gat(node_input, edge_index)

    attn_weights = gat.attn_weights
    if attn_weights is None:
        raise ValueError("❌ Attention weights not stored. Modify GAT forward to save them.")

    # === Convert last world positions → GPS → Pixel
    world_coords = last_pos.cpu().numpy()
    pixel_coords = []

    for x_meter, y_meter in world_coords:
        lat, lon = world_to_gps(x_meter, y_meter)
        u, v = gps_to_pixel(lat, lon)
        pixel_coords.append((u, v))

    pixel_coords = np.array(pixel_coords)  # [N, 2]

    # === Load corresponding video frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"❌ Couldn't read frame {frame_index} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === Normalize attention weights
    attn_weights = attn_weights.mean(dim=1).detach().cpu().numpy()
    attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-6)

    edge_index_np = edge_index.cpu().numpy()
    pos_dict = {i: (pixel_coords[i][0], pixel_coords[i][1]) for i in range(pixel_coords.shape[0])}

    # === Build attention graph
    G = nx.DiGraph()
    G.add_nodes_from(pos_dict.keys())
    edge_colors = []

    for idx, (src, tgt) in enumerate(edge_index_np.T):
        G.add_edge(src, tgt)
        edge_colors.append(attn_weights[idx])

    # === Overlay graph on video frame
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame)

    nx.draw(
        G, pos=pos_dict, edge_color=edge_colors, edge_cmap=plt.cm.plasma,
        node_color='cyan', with_labels=False, node_size=300, width=2,
        arrows=True, ax=ax
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm.set_array(edge_colors)
    plt.colorbar(sm, ax=ax, label="Attention Weight")

    plt.title(f"🧠 GAT Attention Overlay (Frame {frame_index})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
