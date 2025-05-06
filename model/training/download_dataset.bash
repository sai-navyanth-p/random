#!/bin/bash

# Ensure the data directory exists
mkdir -p data

# Download the dataset using Kaggle API
kaggle datasets download -d sumitm004/arxiv-scientific-research-papers-dataset -p data

# Unzip the dataset
unzip -o data/arxiv-scientific-research-papers-dataset.zip -d data/pdfs
