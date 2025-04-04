#!/bin/bash

# Integration script for tensor-house and bptk_py_tutorial repositories
# This script clones both repositories and organizes them into an integrated structure

set -e  # Exit on any error

echo "==================================================================="
echo "         Setting up Integrated AI Modeling Framework"              
echo "==================================================================="

# Create main directory
MAIN_DIR="."  # We're already in the fyp_scm directory
echo "Setting up in current directory: $MAIN_DIR"

# Clone the repositories
echo "\nCloning repositories..."
echo "Cloning tensor-house repository..."
git clone https://github.com/Izaan330/tensor-house.git temp-tensor-house

echo "Cloning bptk_py_tutorial repository..."
git clone https://github.com/Izaan330/bptk_py_tutorial.git temp-bptk_py_tutorial

# Create the directory structure
echo "\nCreating directory structure..."
mkdir -p src/{tensor_house_modules,bptk_modules,integrated}
mkdir -p notebooks/{tensor_house,bptk}
mkdir -p examples/{basic,integrated,advanced}
mkdir -p data/{sample_datasets,simulators}
mkdir -p docs/api

# Create __init__.py files
touch src/__init__.py
touch src/tensor_house_modules/__init__.py
touch src/bptk_modules/__init__.py
touch src/integrated/__init__.py

# Copy TensorHouse notebooks to the integrated structure
echo "\nCopying TensorHouse notebooks..."
mkdir -p notebooks/tensor_house/{pricing,marketing,supply_chain,recommendations,demand_forecasting,smart_manufacturing,search}
cp -r temp-tensor-house/pricing/* notebooks/tensor_house/pricing/ 2>/dev/null || echo "No pricing notebooks found"
cp -r temp-tensor-house/marketing-analytics/* notebooks/tensor_house/marketing/ 2>/dev/null || echo "No marketing notebooks found"
cp -r temp-tensor-house/supply-chain/* notebooks/tensor_house/supply_chain/ 2>/dev/null || echo "No supply chain notebooks found"
cp -r temp-tensor-house/recommendations/* notebooks/tensor_house/recommendations/ 2>/dev/null || echo "No recommendation notebooks found"
cp -r temp-tensor-house/demand-forecasting/* notebooks/tensor_house/demand_forecasting/ 2>/dev/null || echo "No demand forecasting notebooks found"
cp -r temp-tensor-house/smart-manufacturing/* notebooks/tensor_house/smart_manufacturing/ 2>/dev/null || echo "No manufacturing notebooks found"
cp -r temp-tensor-house/search/* notebooks/tensor_house/search/ 2>/dev/null || echo "No search notebooks found"

# Copy BPTK-Py notebooks to the integrated structure
echo "Copying BPTK-Py notebooks..."
mkdir -p notebooks/bptk/{system_dynamics,agent_based,model_library,quickstart}
cp -r temp-bptk_py_tutorial/sd-dsl/* notebooks/bptk/system_dynamics/ 2>/dev/null || echo "No system dynamics notebooks found"
cp -r temp-bptk_py_tutorial/abm/* notebooks/bptk/agent_based/ 2>/dev/null || echo "No agent-based notebooks found"
cp -r temp-bptk_py_tutorial/model_library/* notebooks/bptk/model_library/ 2>/dev/null || echo "No model library notebooks found"
cp -r temp-bptk_py_tutorial/quickstart/* notebooks/bptk/quickstart/ 2>/dev/null || echo "No quickstart notebooks found"

# Copy sample data
echo "Copying sample data..."
mkdir -p data/sample_datasets
cp -r temp-tensor-house/_data/* data/sample_datasets/ 2>/dev/null || echo "No TensorHouse data found"
cp -r temp-bptk_py_tutorial/data/* data/sample_datasets/ 2>/dev/null || echo "No BPTK-Py data found"

# Create basic examples
echo "\nSetting up basic examples..."
mkdir -p examples/basic
cp temp-tensor-house/pricing/price-optimization-multiple-products.ipynb examples/basic/tensor_house_pricing_example.ipynb 2>/dev/null || echo "Pricing example not found"
cp temp-bptk_py_tutorial/quickstart/quickstart.ipynb examples/basic/bptk_quickstart_example.ipynb 2>/dev/null || echo "BPTK quickstart not found"

# Clean up temporary repositories
echo "\nCleaning up..."
rm -rf temp-tensor-house
rm -rf temp-bptk_py_tutorial

echo "\nSetup complete! The integrated structure has been created."
