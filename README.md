# PathoScan

> A local, AI-assisted skin-lesion image screening prototype built with
> EfficientNet-B0 and SmolLM2.

PathoScan accepts a dermoscopic image, classifies it into one of the eight
ISIC 2019 categories, and displays a short educational explanation. Image
classification runs through a FastAPI backend; the optional SmolLM2 model
generates the explanation locally.

## Features

- EfficientNet-B0 inference using the included trained checkpoint
- Eight ISIC 2019 categories: AK, BCC, BKL, DF, MEL, NV, SCC, and VASC
- Optional local SmolLM2-360M-Instruct educational suggestions
- Browser interface with drag-and-drop upload and top-three predictions
- CPU and CUDA support through PyTorch
- No image hosting or third-party inference service

## Project layout

```text
PathoScan_Project/
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── models/
│   └── pathoscan_efficientnet_b0.pth
├── download_smol_model.py
├── requirements.txt
└── README.md
```

The large SmolLM2 weights are intentionally not committed to Git. Running
`download_smol_model.py` downloads them into `models/SmolLM2-360M-Instruct`
when they are needed.

## Quick start on Windows

PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch needs CUDA support, install the matching command from the
[official PyTorch selector](https://pytorch.org/get-started/locally/).
The backend automatically uses CUDA when available and otherwise uses CPU.

### Optional: enable local educational suggestions

```powershell
python download_smol_model.py
```

The download requires internet access and several hundred megabytes of disk
space. Classification still works without SmolLM2; the API uses a cautious
template message as a fallback.

### Start the API

From the project root:

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Check that the classifier is loaded at
[`http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health).

### Start the frontend

Serve the frontend in a second terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500), choose a JPG, PNG, or
WEBP image, and select **Analyze image**.

## API

### `GET /health`

Returns classifier status, optional language-model status, and the active
device.

### `POST /analyze`

Accepts an image upload in the `file` multipart field. Images are limited to
10 MB. The response includes the predicted category, model score, top-three
predictions, description, educational suggestion, and a medical disclaimer.

## Medical and research disclaimer

PathoScan is an educational/research prototype, not a medical device. It is
trained for the ISIC 2019 label categories and is not a general detector for
every skin condition. A classifier score is not a calibrated probability and
does not indicate severity or urgency. The output must not be used to
diagnose, rule out disease, prescribe treatment, or delay professional care.

Seek qualified clinical assessment for a new or changing lesion, or one that
bleeds, hurts, persistently itches, or does not heal.

## Model and dataset notes

- The included `pathoscan_efficientnet_b0.pth` is the project classifier
  checkpoint.
- SmolLM2-360M-Instruct is downloaded from
  [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
  by the supplied script.
- Review the applicable licenses and terms for the models and the ISIC
  dataset before redistribution or deployment.
