#!/bin/bash
# Activate GameLens AI virtual environment
source gamlens_env/bin/activate

# Set environment variables for LightGBM
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

echo "✅ GameLens AI virtual environment activated!"
echo "Environment variables set for LightGBM compatibility."
echo ""
echo "You can now run:"
echo "  - jupyter notebook notebooks/phase1_roas_forecasting.ipynb"
echo "  - streamlit run streamlit_app.py"
echo ""
echo "To deactivate, run: deactivate"
