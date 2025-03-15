import unittest
import torch
import numpy as np
from src.model import ChemFusionSimilarity, calculate_tanimoto_similarity

class TestChemFusionSimilarity(unittest.TestCase):
    
    def setUp(self):
        # Create a small model for testing
        self.descriptor_dim = 12
        self.fp_dim = 16  # Smaller dimension for testing
        self.selfies_dim = 8  # Smaller dimension for testing
        self.model = ChemFusionSimilarity(
            descriptor_dim=self.descriptor_dim,
            fp_dim=self.fp_dim,
            selfies_embed_dim=self.selfies_dim,
            hidden_dim=32
        )
        
        # Create sample inputs
        self.batch_size = 4
        self.selfies_vec = torch.rand(self.batch_size, self.selfies_dim)
        self.fp_vec = torch.rand(self.batch_size, self.fp_dim)
        self.desc_vec = torch.rand(self.batch_size, self.descriptor_dim)
        
    def test_model_forward(self):
        """Test the forward pass of the model"""
        similarity = self.model(self.selfies_vec, self.fp_vec, self.desc_vec)
        
        # Check output shape
        expected_shape = (self.batch_size, 1)
        self.assertEqual(similarity.shape, expected_shape)
        
        # Check output range (should be between 0 and 1 due to sigmoid)
        self.assertTrue(torch.all(similarity >= 0))
        self.assertTrue(torch.all(similarity <= 1))
    
    def test_model_components(self):
        """Test individual components of the model"""
        # Test SELFIES embedding
        selfies_encoded = self.model.selfies_embedding(self.selfies_vec)
        self.assertEqual(selfies_encoded.shape, (self.batch_size, 32))
        
        # Test fingerprint encoder
        fp_encoded = self.model.fp_encoder(self.fp_vec)
        self.assertEqual(fp_encoded.shape, (self.batch_size, 32))
        
        # Test descriptor encoder
        desc_encoded = self.model.descriptor_encoder(self.desc_vec)
        self.assertEqual(desc_encoded.shape, (self.batch_size, 32))
    
    def test_calculate_tanimoto_similarity(self):
        """Test Tanimoto similarity calculation"""
        # Create two identical bit strings (should have similarity 1.0)
        fp1_str = "1010101010101010"  # 16-bit string
        fp2_str = "1010101010101010"  # Identical
        
        similarity = calculate_tanimoto_similarity(fp1_str, fp2_str)
        self.assertEqual(similarity, 1.0)
        
        # Create two completely different bit strings (should have similarity 0.0)
        fp1_str = "1111111100000000"
        fp2_str = "0000000011111111"
        
        similarity = calculate_tanimoto_similarity(fp1_str, fp2_str)
        self.assertEqual(similarity, 0.0)
        
        # Create two partially overlapping bit strings
        fp1_str = "1111000000000000"
        fp2_str = "1100000000000000"
        
        similarity = calculate_tanimoto_similarity(fp1_str, fp2_str)
        self.assertEqual(similarity, 0.5)  # 2 bits in common out of 4 total set bits

if __name__ == '__main__':
    unittest.main()
