import os
import sys
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import torch
import numpy as np

# Ensure the parent directory is in the python path to import the engine
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from inference.engine import InferenceEngine

# ==========================================
# USER CONFIGURATION: ADD YOUR DATASET PATHS
# ==========================================
# Example: 
DATASETS = [
    "/home/teaching/Downloads/grp_29/d1/test",
    "/home/teaching/Downloads/grp_29/d2/test",
    "/home/teaching/Downloads/grp_29/d3/test"
]
# DATASETS = [
#     # Replace these paths with your actual dataset directories
#     "path_to_dataset_1",
#     "path_to_dataset_2",
#     "path_to_dataset_3"
# ]

CHECKPOINT_PATH = project_root / "checkpoints" / "classifier_d1" / "best_model.pth"
BATCH_SIZE = 32

def get_images_and_labels(dataset_path):
    """
    Finds all images in subfolders named 'real' or 'fake'.
    Returns:
        image_paths: list of string paths
        true_labels: list of ints (0 for Real, 1 for Fake)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist.")
        return [], []
    
    image_paths = []
    true_labels = [] 
    
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for sub_dir in dataset_path.iterdir():
        if sub_dir.is_dir():
            dir_name = sub_dir.name.lower()
            if "fake" in dir_name:
                label = 1
            elif "real" in dir_name:
                label = 0
            else:
                continue # Skip unknown folders
                
            for ext in extensions:
                for img_path in sub_dir.glob(f"**/*{ext}"):
                    image_paths.append(str(img_path))
                    true_labels.append(label)
                for img_path in sub_dir.glob(f"**/*{ext.upper()}"):
                    image_paths.append(str(img_path))
                    true_labels.append(label)
                    
    return image_paths, true_labels

def main():
    if not DATASETS or DATASETS[0] == "path_to_dataset_1":
        print("Please edit the DATASETS list in plot.py to add your actual dataset paths.")
        print("Example:")
        print("DATASETS = [")
        print("    '../../d1/test',")
        print("    '../../d2/test',")
        print("]")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading InferenceEngine on {device}...")
    
    chk_path = str(CHECKPOINT_PATH)
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=chk_path,
        device=device
    )
    
    # Store all figures for a combined HTML if needed, or just save them.
    for d_idx, d_path in enumerate(DATASETS):
        print(f"\\n{'='*50}")
        print(f" Evaluating Dataset {d_idx+1}: {d_path}")
        print(f"{'='*50}")
        
        img_paths, y_true = get_images_and_labels(d_path)
        
        if not img_paths:
            print(f"No valid real/fake images found for {d_path}")
            continue
            
        print(f"Found {len(img_paths)} images. Running inference...")
        
        # Batch predict
        preds = engine.predict_batch(img_paths, batch_size=BATCH_SIZE)
        
        y_scores = []
        y_pred = []
        
        # Convert engine predictions to raw scores and binary labels
        for label, conf in preds:
            if label == "Fake":
                prob_fake = conf
                y_pred.append(1)
            else:
                prob_fake = 1.0 - conf
                y_pred.append(0)
            y_scores.append(prob_fake)
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC Curve requires at least one positive and one negative sample
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, float('nan')
        
        print(f"Results for {d_path}:")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  AUC      : {roc_auc:.4f}")
        
        dataset_name = Path(d_path).name
        if not dataset_name or dataset_name == 'test':
            # Use parent name if folder is just 'test'
            dataset_name = Path(d_path).parent.name
        
        # Use absolute path for output within actual_results
        output_dir = Path(__file__).resolve().parent
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                    annot_kws={"size": 14})
        plt.title(f'Confusion Matrix: {dataset_name}\  Accuracy: {acc*100:.2f}%', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        cm_path = output_dir / f'cm_{dataset_name}.png'
        plt.savefig(str(cm_path), dpi=300)
        plt.close()
        
        # Plot ROC curve
        if fpr is not None:
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve: {dataset_name}', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            roc_path = output_dir / f'roc_{dataset_name}.png'
            plt.savefig(str(roc_path), dpi=300)
            plt.close()
        
        print(f"  Saved graphs for {dataset_name} in {output_dir}")

if __name__ == '__main__':
    main()
