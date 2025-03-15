import pandas as pd
import numpy as np
import torch
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from selfies import encoder, decoder

logger = logging.getLogger(__name__)

def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file (if None, log to console only)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained ChemFusionSimilarity model
        test_loader: DataLoader with test data
        device: Device to run evaluation on (if None, use available GPU or CPU)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            selfies_vec, fp_vec, desc_vec, target = [b.to(device) for b in batch]
            output = model(selfies_vec, fp_vec, desc_vec)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(output.cpu().numpy())
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"Model Evaluation Results:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    return results

def plot_correlation(y_true, y_pred, save_path=None):
    """
    Plot correlation between predicted and actual values
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save the plot (if None, display only)
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Similarity')
    plt.ylabel('Predicted Similarity')
    plt.title('Actual vs Predicted Molecular Similarity')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add correlation coefficient
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.annotate(f'r = {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Correlation plot saved to {save_path}")
    else:
        plt.show()

def plot_similarity_heatmap(similarity_matrix, x_labels, y_labels, title, save_path=None):
    """
    Plot similarity matrix as a heatmap
    
    Args:
        similarity_matrix: 2D numpy array of similarity values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
        save_path: Path to save the plot (if None, display only)
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Heatmap saved to {save_path}")
    else:
        plt.show()

def visualize_molecules(smiles_list, labels=None, mol_per_row=4, save_path=None):
    """
    Visualize molecules using RDKit
    
    Args:
        smiles_list: List of SMILES strings
        labels: List of labels for each molecule
        mol_per_row: Number of molecules per row in the grid
        save_path: Path to save the image (if None, return image)
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    if labels is None:
        labels = [f"Mol {i+1}" for i in range(len(mols))]
    
    img = Draw.MolsToGridImage(mols, molsPerRow=mol_per_row, subImgSize=(200, 200), legends=labels)
    
    if save_path:
        img.save(save_path)
        logger.info(f"Molecule visualization saved to {save_path}")
        return None
    else:
        return img

def selfies_to_onehot(selfies_string, vocab_size=64):
    """
    Convert SELFIES string to one-hot encoded vector
    
    Args:
        selfies_string: SELFIES encoded string
        vocab_size: Size of the vocabulary
        
    Returns:
        numpy array: One-hot encoded vector
    """
    # Simple implementation - in practice, you'd need to create a proper vocabulary
    # and handle the actual encoding logic
    import hashlib
    
    vector = np.zeros(vocab_size)
    
    if not selfies_string:
        return vector
    
    # Use a hash function to distribute the characters across the vector
    for char in selfies_string:
        idx = int(hashlib.md5(char.encode()).hexdigest(), 16) % vocab_size
        vector[idx] = 1
    
    return vector

def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    """
    Convert SMILES to Morgan fingerprint
    
    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint
        nBits: Number of bits in fingerprint
        
    Returns:
        numpy array: Binary fingerprint array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(list(fp.ToBitString())).astype(int)
