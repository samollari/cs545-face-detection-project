import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch.nn.functional as F
from dataloaders import LFWPairDataset
from util import parse_lfw_pairs_from_zip
from arcface_model import ArcFaceModel
from collections import OrderedDict

def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

def load_model(model_class, weights_path, device, num_classes):
    model = model_class(num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    clean_state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def lfw_eval_fn(model, device, lfw_loader):
    model.eval()

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

    embeddings1 = torch.cat(embeddings1, dim=0)
    embeddings2 = torch.cat(embeddings2, dim=0)
    labels = torch.cat(labels, dim=0)

    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)

    cosine_sim = (embeddings1 * embeddings2).sum(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Find best threshold
    thresholds = np.arange(-1.0, 1.0, 0.001)
    best_acc = 0
    best_preds = None

    for thresh in thresholds:
        preds = (cosine_sim > thresh).astype(np.int32)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_preds = preds

    return best_preds, labels

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=90)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    zip_path = args.data_zip #"/home/advillatoro/Projects/lfw_edges.zip" # "/home/advillatoro/Projects/lfw.zip"
    csv_name = "pairs.csv"
    image_dir = args.img_dir #"lfw_edges" # "lfw-deepfunneled/lfw-deepfunneled"
    lfw_pairs = parse_lfw_pairs_from_zip(zip_path, csv_name, image_dir)

   # Initialize dataset
    dataset = LFWPairDataset(lfw_pairs, zip_path=zip_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    class_names = ["Mismatch", "Match"]

    # Load model and weights
    model = load_model(ArcFaceModel, args.weights, device, args.num_classes)

    # Get predictions and labels
    y_pred, y_true = lfw_eval_fn(model, device, dataloader)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--data_zip', type=str, required=True, help='Path to validation zip')
    parser.add_argument('--img_dir', type=str, required=True, help='Path inside zip to image folder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, required=True, help='Path to save confusion matrix plot')
    parser.add_argument('--num_classes', type=int, default=4000, help='Number of classes')
    args = parser.parse_args()

    main(args)
