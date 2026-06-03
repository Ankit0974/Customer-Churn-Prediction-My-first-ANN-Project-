# Customer Churn Prediction App

A Machine Learning web application that predicts whether a customer is likely to leave a company (churn) based on customer information. The model is built using **PyTorch**, served through a web interface, containerized with **Docker**, and deployed on **Render**.

---

## 🚀 Live Demo

https://customer-churn-prediction-my-first-ann.onrender.com/

---

## Docker Image Link:
https://hub.docker.com/repository/docker/ankitroy01/customer-churn-app/general


## 📌 Features

- Predict customer churn in real-time
- User-friendly web interface
- Deep Learning model built with PyTorch
- Data preprocessing using StandardScaler
- Dockerized for easy deployment
- Ready for cloud deployment on Render

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Flask
- Docker
- Render

---

## 📂 Project Structure

```bash
CustomerChurn/
│
├── app.py                 # Flask application
├── model.py               # Neural network architecture
├── churn_model2.pth        # Trained PyTorch model
├── scaler2.pkl             # Saved StandardScaler
├── requirements.txt       # Project dependencies
├── Dockerfile             # Docker configuration
└── README.md
```

---

## 📊 Dataset

Dataset used:

-Telco Customer Churn Dataset
-kaggle link:-https://www.kaggle.com/code/basmalaawad/telco-customer-churn-dataset/input

Target Variable:

- **Exited**
  - 0 → Customer stays
  - 1 → Customer leaves

---

## 🧠 Model Architecture

Artificial Neural Network (ANN) implemented using PyTorch.

Architecture:

```text
Input Layer
      ↓
Hidden Layer (ReLU)
      ↓
Hidden Layer (ReLU)
      ↓
Output Layer (Sigmoid)
```

Loss Function:

```python
nn.BCEWithLogitsLoss()
```

Optimizer:

```python
torch.optim.Adam
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Ankit0907/customer-churn-app.git

cd customer-churn-app
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🐳 Docker Setup

Build Docker Image:

```bash
docker build -t customer-churn-app .
```

Run Container:

```bash
docker run -p 5000:5000 customer-churn-app
```

Application will be available at:

```text
http://localhost:5000
```

---

## ☁️ Deployment on Render

1. Push code to GitHub
2. Create a new Web Service on Render
3. Connect GitHub repository
4. Use Docker deployment
5. Deploy

Render automatically builds and hosts the application.

---



## 🔮 Future Improvements

- Add model confidence score
- Explain predictions using SHAP
- User authentication
- Prediction history dashboard
- CI/CD pipeline with GitHub Actions

---

## 👨‍💻 Author

Ankit Roy

B.Tech Computer Science (AI & DS)

GitHub: https://github.com/Ankit097

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use and modify it.
