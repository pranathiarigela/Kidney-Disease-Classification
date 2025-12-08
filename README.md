Kidney Disease Classification

An end-to-end machine learning project that classifies kidney CT scan images into four categories: Normal, Cyst, Tumor and Stone.

The system includes data pipelines, a trained CNN model and a Streamlit app for real-time predictions.

Project Structure
app/                          # Streamlit app

data/raw/                     # Original images

data/processed/               # Preprocessed arrays + splits

models/                       # Trained model + metrics

src/components/               # Ingestion, preprocessing, training, evaluation

src/pipeline/                 # Training and prediction pipelines

utils/                        # Logging, exceptions, helpers

notebooks/EDA.ipynb

main.py

How to Run

1. Setup
   
   python -m venv kidney_env

   kidney_env\Scripts\activate
   
   pip install -r requirements.txt

3. Train the Model

   python main.py

4. Run the Web App
   
   streamlit run app/app.py

Features

   Clean modular structure

   Automated training pipeline

   CNN model with saved weights
   
   Logging and exception handling

   Single-image prediction pipeline

   Streamlit interface for easy use

Outputs

   kidney_model.h5

   label_map.json

   metrics.txt

Preprocessed arrays in data/processed/

Testing

   Run each component independently

   Test full pipeline using main.py

   Test prediction with multiple images through Streamlit
