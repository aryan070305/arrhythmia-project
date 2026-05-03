# 🫀 ECG Arrhythmia Detector

A web application that analyzes ECG (Electrocardiogram) signals to detect cardiac arrhythmias using a deep learning model trained on the MIT-BIH Arrhythmia Database.

---

## ✨ Features

- **Upload ECG data** in CSV/TXT format
- **Auto-detects signal type** — works with both raw continuous recordings and pre-segmented beats
- **R-peak detection** for raw ECG signals using NeuroKit2
- **5-class arrhythmia classification**: Normal, Supraventricular, Ventricular, Fusion, and Unknown beats
- **Detailed results** with confidence scores, beat-by-beat vote breakdown, and waveform preview
- **Custom sample rate** support for different ECG devices

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 or higher** — Download from [python.org](https://www.python.org/downloads/)
  > ⚠️ During installation, make sure to check **"Add Python to PATH"**

### Setup Instructions

**1. Download & extract** the project folder and place it anywhere on your computer (e.g., Desktop).

**2. Open a terminal** inside the `ecg-arrhythmia-detector` folder:

| OS | How to open |
|---|---|
| **Windows** | Open the folder in File Explorer → click the address bar → type `cmd` → press Enter |
| **Mac** | Right-click the folder → "Open Terminal Here" (or open Terminal and `cd` to the folder) |
| **Linux** | Right-click inside the folder → "Open Terminal Here" |

**3. Create a virtual environment:**

```bash
python -m venv venv
```

> If `python` doesn't work, try `python3` instead.

**4. Activate the virtual environment:**

- **Windows (Command Prompt):**
  ```bash
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```bash
  venv\Scripts\Activate.ps1
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` appear at the beginning of your terminal line.

**5. Install dependencies:**

```bash
pip install -r web_app/requirements.txt
```

> This may take a few minutes (TensorFlow is a large package). If it fails, try running `pip install --upgrade pip` first.

**6. Run the application:**

```bash
python web_app/app.py
```

Wait until you see `Model ready.` in the terminal.

**7. Open in your browser:**

Go to **[http://localhost:5000](http://localhost:5000)**

---

## 📁 Project Structure

```
ecg-arrhythmia-detector/
├── model/
│   ├── ecg_arrhythmia_model.h5      # Trained deep learning model
│   └── model_metadata.json           # Class labels & config
├── web_app/
│   ├── app.py                        # Flask backend
│   ├── requirements.txt              # Python dependencies
│   ├── templates/                    # HTML frontend
│   └── uploads/                      # Temporary upload directory
├── kaggle_notebook/                  # Training notebook
└── README.md
```

---

## 📊 Supported Input Formats

| Format | Description |
|---|---|
| **Pre-segmented beat** | A single row of ~187 values (one heartbeat) |
| **Batch of beats** | Multiple heartbeats, each 187 samples, concatenated |
| **Raw ECG recording** | Continuous voltage signal from any ECG device |

- Files should be **CSV** or **TXT** format
- For raw recordings, you can specify your device's **sampling rate** (default: 360 Hz)

---

## 🏷️ Classification Classes

| Class | Label | Severity |
|---|---|---|
| 0 | Normal (N) | ✅ Normal |
| 1 | Supraventricular (S) | ⚠️ Moderate |
| 2 | Ventricular (V) | 🔴 High Risk |
| 3 | Fusion (F) | ⚠️ Moderate |
| 4 | Unknown (Q) | ✅ Normal |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `python` command not found | Use `python3` instead, or reinstall Python with "Add to PATH" checked |
| `pip install` fails | Run `pip install --upgrade pip` first |
| TensorFlow won't install | Make sure you're on Python 3.10–3.12 (TensorFlow doesn't support all versions) |
| Port 5000 already in use | Change the port in `app.py` (last line) or stop the other process |
| Model loading is slow | First launch takes ~30–60 seconds — this is normal |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a certified medical device and should **not** be used for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## 👥 Team

Built as part of an academic project using the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
