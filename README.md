# Radiomics_visualization_Software

Here I created a repository for installing and running the code. I put the instructions step by step:
This software allows you to process, visualize, and analyze radiomics datasets without programming knowledge.

## System Requirements
**Operating System**: Windows, macOS, or Linux
Python Version: 3.8 or later
Required Libraries:
- tkinter
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Pillow

##  Installation Steps
Step 1: Install Python
Download Python from https://www.python.org/downloads/.
During installation:
Check the box: "Add Python to PATH".
Select the Customize installation option and ensure pip is selected.
Verify Python installation:
Open a terminal or command prompt (you can just write cmd in the search box)
Type: python --version
If installed correctly, this shows the installed Python version.

## Step 2: Install Required Libraries
1. Open a terminal or command prompt.
2. Run the following command to install the necessary libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pillow
```
3. Ensure the installation completes without errors.

## Step 3: Save the Software Code
Copy the provided code into a text editor (e.g., Notepad).
Save it as radiomics_analysis.py. Example:
```bash
C:\Users\<YourName>\Documents\radiomics_analysis.py

```

Step 5: Run the Software
1. Open a terminal or command prompt.
2. Navigate to the folder containing radiomics_analysis.py. For example:
```bash
cd C:\Users\<YourName>\Documents
```
3. Run the program:
```bash
python radiomics_analysis.py
```

4. The software will launch with a user-friendly graphical interface.



# How to Use the Software
First make it full size
## Step 1: Load Your Data
Click Browse to select a radiomics dataset (CSV file).
Confirm successful data loading in the popup message.
## Step 2: Configure Settings
Variance Threshold: Remove features with low variation across samples (default: 0.01).
Correlation Threshold: Remove features that are highly correlated (default: 0.9).
Additional Features to Drop: Checkboxes appear if the dataset contains fields like Age, Sex, or BMI.
## Step 3: Analyze Data
Select a Metric for Analysis:
Feature Importance: Rank features by their contribution to survival prediction.
Mutual Information: Identify features most related to survival outcomes.
Click Process Data (*You need to click only once to the Process Data*) you can later change the metric.
## Step 4: Visualize Results
**Show Results**: View a bar chart ranking features by importance or mutual information.
**Show Scatter Plot**: Examine the relationship between the most significant feature and survival time, with a regression line and 
value.
**Show Heatmap**: Visualize the correlation between top features and survival time.
## Step 5: Save Processed Data
Click Save Processed Data to export the dataset after feature selection as a CSV file.



