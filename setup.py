from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="chemfusionsimilarity",
    version="0.1.0",
    author="Benjamin-JHou",
    description="A deep learning model for molecular similarity prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Benjamin-JHou/ChemFusionSimilarity",
    project_urls={
        "Bug Tracker": "https://github.com/Benjamin-JHou/ChemFusionSimilarity/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "process-data=src.data_processing:main",
            "train-model=src.training:main",
        ],
    },
)
