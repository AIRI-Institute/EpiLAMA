conda env create -f environment.yml
conda activate EpiLAMA-IM

# Install MHCnuggets with custom patches 
python -m pip install mhcnuggets
# Find the install directory of mhcnuggets Python module
INSTALL_DIR=$(python -c "import mhcnuggets; import os; print(os.path.dirname(mhcnuggets.__file__))")
# Replace src/models.py with custom models.py
cp custom_data/models.py "$INSTALL_DIR/src/models.py"
# Replace src/predict.py with custom predict.py
cp custom_data/predict.py "$INSTALL_DIR/src/predict.py"

# Install DeepPeptide with custom patches
git clone https://github.com/fteufel/DeepPeptide.git
cp custom_data/utils.py DeepPeptide/predictor/
