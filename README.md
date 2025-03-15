# ChemFusionSimilarity 🧪🔬

> A deep learning-based model for predicting molecular similarity by integrating multiple representation methods

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-%23A00000.svg)](https://www.rdkit.org/)

## Overview 🔍

ChemFusionSimilarity is a deep learning model that predicts similarity between ligands and drug molecules by integrating multiple molecular representation methods:

- **Molecular Descriptors** 📊: Capture macroscopic physicochemical properties
- **Morgan Fingerprints** 👆: Highlight local structural motifs
- **SELFIES Encoding** 🧬: Represent topological structure of molecules

The model incorporates attention mechanisms and feature fusion techniques to leverage the complementary strengths of these representations, overcoming limitations inherent in single-representation approaches.

# Architecture 🏗️

![ChemFusionSimilarity Architecture](https://via.placeholder.com/800x400?text=ChemFusionSimilarity+Architecture)

The architecture includes:
- SELFIES encoder
- Morgan fingerprint encoder
- Molecular descriptor encoder
- Multi-head attention mechanism
- Feature fusion layers

## Installation 💻

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemFusionSimilarity.git
cd ChemFusionSimilarity

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

## Dependencies 📦

- Python 3.8+
- PyTorch
- RDKit
- NumPy
- pandas
- scikit-learn
- SELFIES

## Usage 🚀

### Data Processing

```python
from src.data_processing import process_file

# Process your molecular data
process_file('path/to/your/data.csv', 'processed_data.csv')
```

### Training the Model

```python
from src.model import ChemFusionSimilarity
from src.training import train_model

# Initialize the model
model = ChemFusionSimilarity(descriptor_dim=12)

# Train the model
train_model(model, train_loader, val_loader, epochs=50)
```

### Predicting Similarity

```python
import torch
from src.model import ChemFusionSimilarity

# Load trained model
model = ChemFusionSimilarity(descriptor_dim=12)
model.load_state_dict(torch.load('models/best_cfsim_model.pt'))
model.eval()

# Predict similarity
similarity = model(selfies_vec, fp_vec, desc_vec)
```

## Examples 📝

See the [example notebook](notebooks/examples.ipynb) for detailed usage examples.

## Citation 📄

If you use ChemFusionSimilarity in your research, please cite our work:

```bibtex
@article{author2023chemfusion,
  title={ChemFusionSimilarity: A Deep Learning Approach to Molecular Similarity Prediction},
  author={Author, A.},
  journal={Journal of Cheminformatics},
  year={2023}
}
```

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgements 🙏

- Thanks to all contributors who have helped with this project
- RDKit community for providing excellent cheminformatics tools
- SELFIES developers for the molecular representation format
