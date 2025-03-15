import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """Molecular dataset for training ChemFusionSimilarity model"""
    
    def __init__(self, selfies_data, fp_data, descriptor_data, similarity_scores):
        """
        Args:
            selfies_data: Processed SELFIES encodings
            fp_data: Morgan fingerprint vectors
            descriptor_data: Molecular descriptor values
            similarity_scores: Target similarity scores
        """
        self.selfies_data = torch.FloatTensor(selfies_data)
        self.fp_data = torch.FloatTensor(fp_data)
        self.descriptor_data = torch.FloatTensor(descriptor_data)
        self.similarity_scores = torch.FloatTensor(similarity_scores).view(-1, 1)
        
    def __len__(self):
        return len(self.similarity_scores)
    
    def __getitem__(self, idx):
        return (self.selfies_data[idx], 
                self.fp_data[idx], 
                self.descriptor_data[idx], 
                self.similarity_scores[idx])

def prepare_data(gene_ligands_path, drugbank_path, descriptor_cols, test_size=0.2, batch_size=32):
    """
    Prepare data for training and validation
    
    Args:
        gene_ligands_path: Path to processed gene ligands data
        drugbank_path: Path to processed DrugBank data
        descriptor_cols: List of molecular descriptor column names
        test_size: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    logger.info("Reading processed molecular data...")
    gene_ligands_df = pd.read_csv(gene_ligands_path)
    drugbank_df = pd.read_csv(drugbank_path)
    
    # Generate pairs and calculate similarity scores
    logger.info("Generating molecular pairs and similarity scores...")
    selfies_data = []
    fp_data = []
    descriptor_data = []
    similarity_scores = []
    
    # Sample a subset of pairs to make training feasible
    gene_ligands_sample = gene_ligands_df.sample(min(len(gene_ligands_df), 100))
    drugbank_sample = drugbank_df.sample(min(len(drugbank_df), 1000))
    
    from rdkit import DataStructs
    
    for _, gene_ligands_row in gene_ligands_sample.iterrows():
        for _, drugbank_row in drugbank_sample.iterrows():
            # Process SELFIES (placeholder - actual implementation depends on your encoding strategy)
            # Here assuming you've already converted SELFIES to numeric vectors of size 64
            selfies_vec = np.random.random(64)  # Replace with actual SELFIES vector
            
            # Process Morgan fingerprints
            fp1 = [int(bit) for bit in gene_ligands_row['Morgan_FP']]
            fp2 = [int(bit) for bit in drugbank_row['Morgan_FP']]
            fp_vec = np.array([a ^ b for a, b in zip(fp1, fp2)])  # XOR fingerprints
            
            # Process descriptors
            desc1 = gene_ligands_row[descriptor_cols].values
            desc2 = drugbank_row[descriptor_cols].values
            desc_vec = np.abs(desc1 - desc2)  # Absolute difference of descriptors
            
            # Calculate Tanimoto similarity as target
            fp1_bit = DataStructs.CreateFromBitString(gene_ligands_row['Morgan_FP'])
            fp2_bit = DataStructs.CreateFromBitString(drugbank_row['Morgan_FP'])
            tanimoto = DataStructs.TanimotoSimilarity(fp1_bit, fp2_bit)
            
            selfies_data.append(selfies_vec)
            fp_data.append(fp_vec)
            descriptor_data.append(desc_vec)
            similarity_scores.append(tanimoto)
    
    # Convert to numpy arrays
    selfies_data = np.array(selfies_data)
    fp_data = np.array(fp_data)
    descriptor_data = np.array(descriptor_data)
    similarity_scores = np.array(similarity_scores)
    
    # Standardize descriptor data
    scaler = StandardScaler()
    descriptor_data = scaler.fit_transform(descriptor_data)
    
    # Split data
    indices = np.arange(len(similarity_scores))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = MolecularDataset(
        selfies_data[train_indices],
        fp_data[train_indices],
        descriptor_data[train_indices],
        similarity_scores[train_indices]
    )
    
    val_dataset = MolecularDataset(
        selfies_data[val_indices],
        fp_data[val_indices],
        descriptor_data[val_indices],
        similarity_scores[val_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logger.info(f"Created training set with {len(train_dataset)} samples")
    logger.info(f"Created validation set with {len(val_dataset)} samples")
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, model_dir='models'):
    """
    Train the ChemFusionSimilarity model
    
    Args:
        model: ChemFusionSimilarity model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        model_dir: Directory to save model checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            selfies_vec, fp_vec, desc_vec, target = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            output = model(selfies_vec, fp_vec, desc_vec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                selfies_vec, fp_vec, desc_vec, target = [b.to(device) for b in batch]
                output = model(selfies_vec, fp_vec, desc_vec)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_cfsim_model.pt'))
            logger.info(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_cfsim_model.pt'))
    logger.info("Training completed!")
    
    return model

def calculate_similarity_matrix(model, gene_ligands_df, drugbank_df, descriptor_cols, model_path=None, batch_size=32):
    """
    Calculate similarity matrix between gene ligands and DrugBank molecules
    
    Args:
        model: ChemFusionSimilarity model instance
        gene_ligands_df: DataFrame with gene ligands data
        drugbank_df: DataFrame with DrugBank data
        descriptor_cols: List of molecular descriptor column names
        model_path: Path to load model weights (if None, use provided model)
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with similarity scores for each molecule pair
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if path provided
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    model.eval()
    
    # Initialize arrays for results
    n_gene_ligands = len(gene_ligands_df)
    n_drugbank = len(drugbank_df)
    tanimoto_matrix = np.zeros((n_gene_ligands, n_drugbank))
    cfsim_matrix = np.zeros((n_gene_ligands, n_drugbank))
    
    from rdkit import DataStructs
    
    # Process in batches to avoid memory issues
    logger.info("Calculating similarity matrices...")
    
    with torch.no_grad():
        for i in range(0, n_gene_ligands, 10):
            for j in range(0, n_drugbank, batch_size):
                # Get batch of molecules
                gene_ligands_batch = gene_ligands_df.iloc[i:min(i+10, n_gene_ligands)]
                drugbank_batch = drugbank_df.iloc[j:min(j+batch_size, n_drugbank)]
                
                # Prepare batch data
                batch_data = []
                indices = []
                
                for idx1, (_, gene_ligands_row) in enumerate(gene_ligands_batch.iterrows()):
                    for idx2, (_, drugbank_row) in enumerate(drugbank_batch.iterrows()):
                        # Similar to prepare_data function
                        selfies_vec = np.random.random(64)  # Replace with actual SELFIES vector
                        
                        fp1 = [int(bit) for bit in gene_ligands_row['Morgan_FP']]
                        fp2 = [int(bit) for bit in drugbank_row['Morgan_FP']]
                        fp_vec = np.array([a ^ b for a, b in zip(fp1, fp2)])
                        
                        desc1 = gene_ligands_row[descriptor_cols].values
                        desc2 = drugbank_row[descriptor_cols].values
                        desc_vec = np.abs(desc1 - desc2)
                        
                        # Calculate Tanimoto similarity
                        fp1_bit = DataStructs.CreateFromBitString(gene_ligands_row['Morgan_FP'])
                        fp2_bit = DataStructs.CreateFromBitString(drugbank_row['Morgan_FP'])
                        tanimoto = DataStructs.TanimotoSimilarity(fp1_bit, fp2_bit)
                        
                        batch_data.append((selfies_vec, fp_vec, desc_vec))
                        indices.append((i+idx1, j+idx2))
                        tanimoto_matrix[i+idx1, j+idx2] = tanimoto
                
                if not batch_data:
                    continue
                    
                # Process batch with model
                selfies_batch = torch.FloatTensor([item[0] for item in batch_data]).to(device)
                fp_batch = torch.FloatTensor([item[1] for item in batch_data]).to(device)
                desc_batch = torch.FloatTensor([item[2] for item in batch_data]).to(device)
                
                predictions = model(selfies_batch, fp_batch, desc_batch)
                
                # Store predictions
                for idx, (i_idx, j_idx) in enumerate(indices):
                    cfsim_matrix[i_idx, j_idx] = predictions[idx].item()
                
                logger.info(f"Processed batch: Gene_Ligands {i}-{min(i+10, n_gene_ligands)}, DrugBank {j}-{min(j+batch_size, n_drugbank)}")
    
    # Create result DataFrame
    result_data = []
    for i in range(n_gene_ligands):
        for j in range(n_drugbank):
            result_data.append({
                'Gene_Ligands_ID': gene_ligands_df.iloc[i].get('ID', f'Gene_Ligands_{i}'),
                'DrugBank_ID': drugbank_df.iloc[j].get('ID', f'DrugBank_{j}'),
                'Gene_Ligands_SMILES': gene_ligands_df.iloc[i]['Standardized_SMILES'],
                'DrugBank_SMILES': drugbank_df.iloc[j]['Standardized_SMILES'],
                'Tanimoto_Similarity': tanimoto_matrix[i, j],
                'CFSim_Similarity': cfsim_matrix[i, j]
            })
    
    result_df = pd.DataFrame(result_data)
    logger.info("Similarity calculation completed!")
    
    return result_df, tanimoto_matrix, cfsim_matrix
