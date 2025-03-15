import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from selfies import encoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_smiles(smiles):
    """Standardize SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def generate_selfies(smiles):
    """Generate SELFIES encoding"""
    try:
        return encoder(smiles)
    except:
        return None

def calculate_descriptors(mol):
    """Calculate molecular descriptors"""
    if mol is None:
        return None
        
    try:
        return {
            'MolWeight': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumRings': Descriptors.RingCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'MolMR': Descriptors.MolMR(mol)
        }
    except:
        return None

def generate_morgan_fingerprint(mol, radius=2, nBits=1024):
    """Generate Morgan fingerprint"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        return fp.ToBitString()
    except:
        return None

def process_file(input_file, output_file):
    """Process a single file"""
    logger.info(f"Processing file: {input_file}")
    
    # Read data
    df = pd.read_csv(input_file)
    logger.info(f"Successfully read {len(df)} records")
    
    # Standardize SMILES
    df['Standardized_SMILES'] = df['SMILES'].apply(standardize_smiles)
    valid_smiles_mask = df['Standardized_SMILES'].notna()
    df = df[valid_smiles_mask]
    logger.info(f"Number of valid SMILES: {len(df)}")
    
    # Create molecule objects
    mols = [Chem.MolFromSmiles(s) for s in df['Standardized_SMILES']]
    
    # Generate SELFIES
    logger.info("Generating SELFIES encoding...")
    df['SELFIES'] = df['Standardized_SMILES'].apply(generate_selfies)
    
    # Calculate molecular descriptors
    logger.info("Calculating molecular descriptors...")
    descriptors = [calculate_descriptors(mol) for mol in mols]
    descriptors_df = pd.DataFrame(descriptors)
    
    # Generate Morgan fingerprints
    logger.info("Generating Morgan fingerprints...")
    fingerprints = [generate_morgan_fingerprint(mol) for mol in mols]
    df['Morgan_FP'] = fingerprints
    
    # Merge all data
    result_df = pd.concat([df, descriptors_df], axis=1)
    
    # Save results
    logger.info(f"Saving processed data to: {output_file}")
    result_df.to_csv(output_file, index=False)
    
    return result_df

def main():
    # Process two files
    logger.info("Processing gene_ligands data...")
    folh1_df = process_file('data/gene_ligands.csv', 'data/gene_ligands_new.csv')
    
    logger.info("Processing DrugBank data...")
    drugbank_df = process_file('data/drugbank_with_smiles.csv', 'data/drugbank_new.csv')
    
    logger.info("Data processing completed!")

if __name__ == "__main__":
    main()
