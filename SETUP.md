🔹 Requirements

Make sure the following are installed on your PC:

Python 3.9 – 3.11
👉 https://www.python.org/downloads/

Git (optional, but recommended)
👉 https://git-scm.com/downloads

Check installation:

python --version
pip --version

🔹 Step 1: Clone the Repository

git clone https://github.com/RagibH/DIP-Project.git
cd DIP-Project

🔹 Step 2: Create Virtual Environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Run the Application
python -m streamlit run app.py


The app will open automatically in your browser:

http://localhost:8501

🔹 Troubleshooting

If cv2 error occurs → make sure opencv-python-headless is installed

If Streamlit command fails → use python -m streamlit

🔹 Project Structure
DIP-Project/
├── app.py
├── requirements.txt
├── README.md
└── venv/ (created locally, not included in GitHub)