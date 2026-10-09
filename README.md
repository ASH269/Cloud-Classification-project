# CloudScope

A small Flask website for a cloud-image classification project.

## What works now

- The page opens and lets you select and preview an image.
- Flask reports whether a trained model file is available.
- The app does not invent predictions when model weights are missing.

## Run on Windows

Use Python 3.10 or newer. Open this folder in VS Code and run these commands in the terminal:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The default install contains Flask only, so you do not need TensorFlow to view the website.

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python` for the install and run commands.

## What still needs a model

Real predictions and evaluation results require trained weights saved as `model/cloud_classifier_model.keras` or `model/cloud_classifier_model.h5`. When you have a model, install the additional packages with:

```powershell
python -m pip install tensorflow pillow
```

The model must accept RGB images sized 150×150 with pixel values from 0 to 1, and return 11 scores. The class order in `CLASS_NAMES` in `app.py` must match the order used during training. Check the dataset's labels and evaluate on held-out images before reporting performance.

## Project files

```text
app.py
requirements.txt
README.md
model/
templates/index.html
static/app.js
static/style.css
```
