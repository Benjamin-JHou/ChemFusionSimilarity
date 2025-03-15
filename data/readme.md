# Data Directory 📊 🧪

This directory contains raw data files for the ChemFusionSimilarity project. These datasets are used for training and evaluating the molecular similarity model.

## Available Datasets

### 1. FOLH1.csv
- **Description**: Contains SMILES representations of FOLH1 (Folate Hydrolase 1) ligands
- **Format**: CSV file with SMILES strings and additional properties
- **Usage**: Used as ligand molecules for similarity comparison

### 2. drugbank_with_smiles.csv
- **Description**: A subset of DrugBank database containing approved drugs with their SMILES representations
- **Format**: CSV file containing drug IDs, names, SMILES, and other properties
- **Usage**: Used as target drug molecules for similarity assessment

## Processed Data

After running the data processing scripts, the following files will be generated:

- **FOLH1_new.csv**: Processed FOLH1 data with standardized SMILES, SELFIES, molecular descriptors, and Morgan fingerprints
- **drugbank_new.csv**: Processed DrugBank data with standardized SMILES, SELFIES, molecular descriptors, and Morgan fingerprints


### Processed Data Fields
- **Standardized_SMILES**: Canonicalized SMILES strings
- **SELFIES**: Self-referencing embedded strings representation
- **Morgan_FP**: 1024-bit Morgan fingerprints (radius 2)
- **MolWeight**: Molecular weight
- **LogP**: Octanol-water partition coefficient
- **TPSA**: Topological polar surface area
- **NumRotatableBonds**: Number of rotatable bonds
- **NumHAcceptors**: Number of hydrogen bond acceptors
- **NumHDonors**: Number of hydrogen bond donors
- **NumAromaticRings**: Number of aromatic rings
- **NumRings**: Total number of rings
- **FractionCSP3**: Fraction of carbon atoms with sp3 hybridization
- **NumAliphaticRings**: Number of aliphatic rings
- **BertzCT**: Bertz complexity index
- **MolMR**: Molecular refractivity

## Data Preprocessing

To process the raw data files, run:

```bash
python -m src.data_processing
```

This will generate the processed files with additional molecular features needed for the model.

## Notes 📝

- Make sure to cite the original data sources if you use these datasets in your research
- The raw data files are not modified and are kept for reference
- Some molecules may be filtered out during processing if they have invalid SMILES strings
