## Generating the EDA Report

Before running the script, ensure you have the required dependencies installed. You can do this by:

### Option 1: Using Conda
1. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate pytf
   ```
2. Run the script to generate the EDA report:
   ```bash
   bash generate_eda.sh
   ```

### Option 2: Using pip
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script to generate the EDA report:
   ```bash
   bash generate_eda.sh
   ```

### Output
The script will execute the `EDA.ipynb` notebook and create an HTML report named `EDA_Report.html` in the current directory.

### Notes
- Ensure that Jupyter Notebook and `nbconvert` are installed, as they are required to execute the notebook and generate the HTML report.
- If you encounter any issues with the setup or execution, double-check that all dependencies are installed correctly.
