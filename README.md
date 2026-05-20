SentinelAI
-----------
A lightweight machine learning-based phishing URL and spam message detection system built using Python and Streamlit.

SentinelAI analyzes:
- SMS messages
- Emails
- URLs

to estimate whether content may be spam, phishing, or potentially unsafe.

The project combines:
- Machine Learning models
- Text analysis
- URL feature extraction
- Rule-based phishing detection

into a simple interactive web application.

Features
---------
- SMS / Email spam detection
- Phishing URL analysis
- Risk scoring system
- Interactive Streamlit dashboard
- TF-IDF + Logistic Regression for message classification
- Random Forest model for URL analysis
- Hybrid ML + heuristic detection approach

Tech Stack
-----------
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly

Installation
--------------
Clone the repository:

git clone https://github.com/your-username/SentinelAI.git
cd SentinelAI

Create a virtual environment:  python -m venv venv

Activate the environment:
Windows:  venv\Scripts\activate
Linux / Mac:  source venv/bin/activate

Install dependencies:  pip install -r requirements.txt

Training the Models
--------------------
Run:  python -m training.train_all

This generates:
- sms_pipeline.joblib
- url_pipeline.joblib

inside the models/ folder.

Running the Application
------------------------
streamlit run app/streamlit_app.py

Notes
-------
- This project is intended for educational and learning purposes.
- Predictions are probabilistic and may not always be fully accurate.
- The system combines machine learning predictions with rule-based heuristics for improved detection.