# AI-Based Student Performance Analysis and Prediction

## Project status
This is a working educational demonstration built with **synthetic (fictional) data**. It is not evidence that this project was completed during an internship. Do not describe it as internship work unless that is factually true and supported by your records.

## What it does
- Loads a CSV dataset.
- Validates required columns and value ranges.
- Checks missing values and duplicate rows.
- Displays descriptive statistics.
- Creates two exploratory charts.
- Creates Low / Medium / High categories from final scores.
- Trains and evaluates a Random Forest baseline classifier.
- Saves a confusion-matrix chart.

## Requirements
Python 3.10 or newer is recommended.

## Run in VS Code (Windows)
1. Extract this ZIP file.
2. Open the `Student_Performance_Analysis_Project` folder in VS Code.
3. Open Terminal -> New Terminal.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt:
   ```cmd
   .venv\Scripts\activate.bat
   ```
6. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```
7. Run:
   ```bash
   python main.py
   ```

## Files
- `main.py` - project source code
- `student_performance.csv` - synthetic dataset
- `requirements.txt` - Python dependencies
- `outputs/` - charts generated after running the program

## Dataset columns
- `attendance_pct`: attendance percentage (0-100)
- `assignment_pct`: assignment completion percentage (0-100)
- `practice_hours`: weekly practice hours
- `previous_score`: previous assessment score (0-100)
- `final_score`: synthetic final assessment score (0-100)

## Category thresholds
- Low: below 50
- Medium: 50 to below 75
- High: 75 to 100

## Important limitations
The dataset is artificial and the score formula is illustrative. Model accuracy only measures performance on this generated sample and must not be interpreted as real-world educational accuracy. The model must not be used for grading, admissions, discipline, or decisions about real learners.
