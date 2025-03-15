import unittest
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from tempfile import NamedTemporaryFile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    standardize_smiles, 
    generate_selfies, 
    calculate_descriptors, 
    generate_morgan_fingerprint, 
    process_file
)

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_smiles = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CCO',  # Ethanol
            'C1=CC=C2C(=C1)C=CC=C2',  # Naphthalene
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'C1=CC=CC=C1',  # Benzene
            'INVALID_SMILES'  # Invalid SMILES
        ]
        
        # Create a temporary CSV file for testing
        self.temp_csv_file = NamedTemporaryFile(suffix='.csv', delete=False)
        df = pd.DataFrame({
            'SMILES': self.test_smiles,
            'Name': ['Aspirin', 'Ethanol', 'Naphthalene', 'Caffeine', 'Benzene', 'Invalid']
        })
        df.to_csv(self.temp_csv_file.name, index=False)
        
    def tearDown(self):
        """Clean up test data"""
        os.unlink(self.temp_csv_file.name)
    
    def test_standardize_smiles(self):
        """Test SMILES standardization"""
        standardized = [standardize_smiles(smiles) for smiles in self.test_smiles]
        
        # Check valid SMILES are standardized
        self.assertIsNotNone(standardized[0])
        self.assertIsNotNone(standardized[1])
        self.assertIsNotNone(standardized[2])
        self.assertIsNotNone(standardized[3])
        self.assertIsNotNone(standardized[4])
        
        # Check invalid SMILES returns None
        self.assertIsNone(standardized[5])
    
    def test_generate_selfies(self):
        """Test SELFIES generation"""
        standardized = [standardize_smiles(smiles) for smiles in self.test_smiles]
        selfies = [generate_selfies(smiles) for smiles in standardized if smiles is not None]
        
        # Check we got SELFIES for valid SMILES
        self.assertEqual(len(selfies), 5)
        self.assertIsInstance(selfies[0], str)
    
    def test_calculate_descriptors(self):
        """Test descriptor calculation"""
        mols = [Chem.MolFromSmiles(smiles) for smiles in self.test_smiles]
        descriptors = [calculate_descriptors(mol) for mol in mols]
        
        # Check descriptor calculation for valid molecules
        self.assertIsNotNone(descriptors[0])
        self.assertIsInstance(descriptors[0], dict)
        self.assertIn('MolWeight', descriptors[0])
        self.assertIn('LogP', descriptors[0])
        self.assertIn('TPSA', descriptors[0])
        
        # Check descriptor calculation for invalid molecule
        self.assertIsNone(descriptors[5])
    
    def test_generate_morgan_fingerprint(self):
        """Test Morgan fingerprint generation"""
        mols = [Chem.MolFromSmiles(smiles) for smiles in self.test_smiles]
        fingerprints = [generate_morgan_fingerprint(mol) for mol in mols]
        
        # Check fingerprint generation for valid molecules
        self.assertIsNotNone(fingerprints[0])
        self.assertIsInstance(fingerprints[0], str)
        self.assertEqual(len(fingerprints[0]), 1024)  # Default nBits=1024
        
        # Check fingerprint generation for invalid molecule
        self.assertIsNone(fingerprints[5])
    
    def test_process_file(self):
        """Test file processing"""
        output_file = NamedTemporaryFile(suffix='.csv', delete=False)
        try:
            result_df = process_file(self.temp_csv_file.name, output_file.name)
            
            # Check result DataFrame
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertGreater(len(result_df), 0)
            self.assertIn('Standardized_SMILES', result_df.columns)
            self.assertIn('SELFIES', result_df.columns)
            self.assertIn('Morgan_FP', result_df.columns)
            self.assertIn('MolWeight', result_df.columns)
            
            # Check output file exists and contains data
            self.assertTrue(os.path.exists(output_file.name))
            output_df = pd.read_csv(output_file.name)
            self.assertEqual(len(output_df), len(result_df))
            
        finally:
            os.unlink(output_file.name)
            
if __name__ == '__main__':
    unittest.main()
