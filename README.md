# EcoGuard AI 🌍🛡️

**EcoGuard AI** is a comprehensive, AI-powered computer vision platform designed to tackle industrial pollution, enforce worker safety compliance, and monitor environmental health. 

Developed at **ESPRIT (École Supérieure Privée d'Ingénierie et de Technologies / Esprit School of Engineering)**, EcoGuard utilizes advanced deep learning architectures (YOLOv8 & PyTorch U-Net) and is deployed via a modern Django web application. It provides real-time, interactive insights into high-risk industrial zones (like mining operations) and their surrounding ecosystems.

---

## ✨ Key Features

### 👷‍♂️ Industrial Worker Safety
* **PPE Compliance Tracking:** Real-time YOLOv8 models fine-tuned to detect hard hats, medical masks, and heavy-duty gas masks.
* **Explainable AI (XAI):** EigenCAM integrations to visualize exactly *what* the model is looking at when making safety predictions.

### 🏭 Environmental & Mining Monitoring
* **Mining Area Segmentation:** PyTorch U-Net models trained on satellite imagery to precisely calculate the footprint of legal and illegal mining operations.
* **Soil Health Analysis:** The custom "Ghada" U-Net model evaluates land degradation and soil health based on the proximity and density of mining excavations.
* **Deforestation Tracking:** Automated tracking of forest loss around industrial zones using custom satellite segmentation.
* **Smoke & Fire Detection:** Early warning systems for industrial exhaust and wildfires.

### 🐟 Wildlife & Ecosystem Tracking
* **Aquatic Contamination Analysis:** Advanced fish behavior tracking. The system uses Re-Identification (Re-ID) and trajectory mapping to classify fish swimming patterns (Normal vs. Stressed) as an early indicator of phosphogypsum water contamination.
* **Terrestrial Wildlife:** YOLO-based animal detection to monitor wildlife displacement near mining zones.

### 🔐 Smart Authentication & Administration
* **OCR-Powered Registration:** Employees register by uploading their ID cards. The system uses `EasyOCR` to automatically extract their First and Last names, auto-generate a corporate email (`first_last@EcoGuard.ai`), and generate a highly secure random password.
* **Approval Workflow:** New accounts are placed in a "Pending Approval" state.
* **Admin Dashboard:** Superusers have access to a User Management interface to view ID cards and securely Approve, Reject, or Delete employee access.

---

## 🛠️ Technology Stack

**Machine Learning & Computer Vision:**
* `PyTorch` / `Torchvision`
* `Ultralytics YOLOv8`
* `Segmentation Models PyTorch (SMP)`
* `OpenCV` / `NumPy`
* `EasyOCR`

**Web Development:**
* `Django` (Python Web Framework)
* `SQLite3` (Database)
* `Bootstrap 5` (Mazer Admin Template)
* `Chart.js` / `ApexCharts` (Data Visualization)

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zaabola/Projet_Mining_Pollution.git
   cd Projet_Mining_Pollution
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   *(Ensure you have PyTorch installed according to your CUDA version first)*
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Database Migrations:**
   ```bash
   cd web_app
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create an Admin User:**
   ```bash
   python manage.py createsuperuser
   ```

6. **Start the Development Server:**
   ```bash
   python manage.py runserver
   ```
   Access the dashboard at `http://localhost:8000`.

---

## 📸 Screenshots

*(Add screenshots of your UI, Dark Mode, Mining Segmentation, and Fish Tracking here!)*

---

## 📄 Credits & Academic Context
This project was developed at **ESPRIT (Esprit School of Engineering)** as part of an advanced environmental technology and artificial intelligence initiative. All rights reserved.
