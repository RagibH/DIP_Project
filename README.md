# Digital Image Processing (DIP) Project

This project is a Digital Image Processing (DIP) application built using Python and Streamlit. It provides an interactive user interface for applying and visualizing different image processing techniques.

## 🔹 Requirements

Make sure the following are installed on your PC:

- Python 3.9 – 3.11  
  https://www.python.org/downloads/

- Git (optional but recommended)  
  https://git-scm.com/downloads

Check installation:

python --version  
pip --version

## 🔹 Step 1: Clone the Repository

git clone https://github.com/RagibH/DIP-Project.git  
cd DIP-Project

(Alternatively, download the ZIP file and extract it.)

## 🔹 Step 2: Create Virtual Environment

python -m venv venv

Activate it:

Windows:  
venv\Scripts\activate

Linux / macOS:  
source venv/bin/activate

## 🔹 Step 3: Install Dependencies

pip install -r requirements.txt

## 🔹 Step 4: Run the Application

python -m streamlit run app.py

The application will open automatically in your browser at:

http://localhost:8501

## 🔹 Troubleshooting

If a cv2 error occurs, make sure opencv-python-headless is installed.

If the Streamlit command fails, always use python -m streamlit.

## 🔹 Project Structure

DIP-Project/  
├── app.py  
├── requirements.txt  
├── README.md  
└── venv/ (created locally, not included in GitHub)

## ✅ Notes

- The venv folder is created locally and should not be uploaded to GitHub  
- Works on Windows, Linux, and macOS  
- Ready for deployment on Render or Streamlit Cloud
