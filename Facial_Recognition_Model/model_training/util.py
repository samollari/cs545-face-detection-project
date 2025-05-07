import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import csv
import torch.distributed as dist
import zipfile
import io

def parse_lfw_pairs_from_zip(zip_path, csv_inside_zip, images_dir_inside_zip):
    """
    Parse the LFW pairs CSV from inside a zip file into (path1, path2, label) triples.

    Args:
        zip_path (str): Path to the zip file containing pairs.csv and images.
        csv_inside_zip (str): Path inside the zip to the pairs.csv file.
        images_dir_inside_zip (str): Path inside the zip to the image root directory.

    Returns:
        list of (path1_in_zip, path2_in_zip, label) tuples
    """
    pairs = []

    with zipfile.ZipFile(zip_path, 'r') as archive:
        with archive.open(csv_inside_zip) as f:
            decoded = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(decoded)
            rows = list(reader)

            fold_size = 600

            for fold_start in range(1, len(rows), fold_size):
                fold_rows = rows[fold_start:fold_start + fold_size]

                # First 300 rows: positive pairs
                for row in fold_rows[:300]:
                    name = row[0]
                    img1_idx = int(row[1])
                    img2_idx = int(row[2])

                    img1_path = os.path.join(images_dir_inside_zip, name, f"{name}_{img1_idx:04d}.jpg")
                    img2_path = os.path.join(images_dir_inside_zip, name, f"{name}_{img2_idx:04d}.jpg")

                    pairs.append((img1_path, img2_path, 1))  # Positive

                # Next 300 rows: negative pairs
                for row in fold_rows[300:]:
                    name1 = row[0]
                    img1_idx = int(row[1])
                    name2 = row[2]
                    img2_idx = int(row[3])

                    img1_path = os.path.join(images_dir_inside_zip, name1, f"{name1}_{img1_idx:04d}.jpg")
                    img2_path = os.path.join(images_dir_inside_zip, name2, f"{name2}_{img2_idx:04d}.jpg")

                    pairs.append((img1_path, img2_path, 0))  # Negative

    return pairs

@torch.no_grad()
def lfw_eval_fn(model, device, lfw_loader, world_size, use_dist, rank):
    model.eval()

    # Collect embeddings and labels across all GPUs
    embeddings1 = []
    embeddings2 = []
    labels = []

    for img1, img2, label in lfw_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        emb1 = model(img1)
        emb2 = model(img2)

        embeddings1.append(emb1)
        embeddings2.append(emb2)
        labels.append(label)

    # Concatenate embeddings and labels across all GPUs
    embeddings1 = torch.cat(embeddings1, dim=0)
    embeddings2 = torch.cat(embeddings2, dim=0)
    labels = torch.cat(labels, dim=0)

    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)

    if use_dist:
        # Gather results from all GPUs
        all_embeddings1 = [torch.zeros_like(embeddings1) for _ in range(world_size)]
        all_embeddings2 = [torch.zeros_like(embeddings2) for _ in range(world_size)]
        all_labels = [torch.zeros_like(labels) for _ in range(world_size)]

        dist.all_gather(all_embeddings1, embeddings1)
        dist.all_gather(all_embeddings2, embeddings2)
        dist.all_gather(all_labels, labels)

        # Flatten the list of tensors into a single tensor for all GPUs
        embeddings1 = torch.cat(all_embeddings1, dim=0)
        embeddings2 = torch.cat(all_embeddings2, dim=0)
        labels = torch.cat(all_labels, dim=0)

    # Compute cosine similarity
    cosine_sim = (embeddings1 * embeddings2).sum(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Find best threshold
    thresholds = np.arange(-1.0, 1.0, 0.001)
    accuracies = []

    for thresh in thresholds:
        preds = (cosine_sim > thresh).astype(np.int32)
        acc = (preds == labels).mean()
        accuracies.append(acc)

    best_idx = np.argmax(accuracies)
    best_thresh = thresholds[best_idx]
    best_acc = accuracies[best_idx]

    if rank == 0 or (not use_dist):  # Only the rank 0 process prints the final result
        print(f"[LFW] Best Threshold: {best_thresh:.4f} | Accuracy: {best_acc * 100:.2f}%")

    return best_acc * 100.0  # Return the best accuracy as percentage

@torch.no_grad()
def visualize_embeddings(model, data_loader, device, num_classes_to_show=10):
    model.eval()

    embeddings = []
    labels = []

    for images, lbls in data_loader:
        images = images.to(device)
        lbls = lbls.to(device)

        emb = model(images)  # No labels in forward
        embeddings.append(emb)
        labels.append(lbls)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Subsample classes
    selected_indices = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            selected_indices.extend(idx[:min(10, len(idx))])  # up to 10 samples per class

    embeddings = embeddings[selected_indices]
    labels = labels[selected_indices]

    # t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab20", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE of Embeddings")
    plt.show()