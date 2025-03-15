import torch
import torch.nn as nn
import numpy as np

class ChemFusionSimilarity(nn.Module):
    def __init__(self, descriptor_dim, fp_dim=1024, selfies_embed_dim=64, hidden_dim=256):
        """
        ChemFusionSimilarity model for predicting molecular similarity
        
        Args:
            descriptor_dim (int): Dimension of molecular descriptors
            fp_dim (int): Dimension of fingerprint vectors (default: 1024)
            selfies_embed_dim (int): Dimension of SELFIES embedding (default: 64)
            hidden_dim (int): Dimension of hidden layers (default: 256)
        """
        super(ChemFusionSimilarity, self).__init__()
        
        # SELFIES encoder
        self.selfies_embedding = nn.Linear(selfies_embed_dim, hidden_dim)
        
        # Morgan fingerprint encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Molecular descriptor encoder
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, selfies_vec, fp_vec, desc_vec):
        """
        Forward pass of the model
        
        Args:
            selfies_vec: Tensor of SELFIES vectors
            fp_vec: Tensor of Morgan fingerprint vectors
            desc_vec: Tensor of molecular descriptor vectors
            
        Returns:
            Tensor: Predicted similarity scores
        """
        # Encode features
        selfies_encoded = self.selfies_embedding(selfies_vec)
        fp_encoded = self.fp_encoder(fp_vec)
        desc_encoded = self.descriptor_encoder(desc_vec)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(
            selfies_encoded.unsqueeze(0),
            torch.cat([fp_encoded.unsqueeze(0), desc_encoded.unsqueeze(0)]),
            torch.cat([fp_encoded.unsqueeze(0), desc_encoded.unsqueeze(0)])
        )
        
        # Feature fusion
        combined = torch.cat([
            attn_output.squeeze(0),
            fp_encoded,
            desc_encoded
        ], dim=1)
        
        similarity = self.fusion_layer(combined)
        return similarity

def calculate_tanimoto_similarity(fp1_str, fp2_str):
    """
    Calculate Tanimoto similarity
    
    Args:
        fp1_str: Binary string representation of first fingerprint
        fp2_str: Binary string representation of second fingerprint
        
    Returns:
        float: Tanimoto similarity score
    """
    from rdkit import DataStructs
    fp1 = DataStructs.CreateFromBitString(fp1_str)
    fp2 = DataStructs.CreateFromBitString(fp2_str)
    return DataStructs.TanimotoSimilarity(fp1, fp2)
