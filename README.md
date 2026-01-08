# OCR Tifinagh Script

<div align="center">

**A CNN-based Optical Character Recognition system for Tifinagh script**

</div>

## 📖 About

This project implements a deep learning-based Optical Character Recognition (OCR) system specifically designed for **Tifinagh script**, the writing system used for Amazigh (Berber) languages. The model can extract text from images, scanned documents, and handwritten content, converting Tifinagh characters into a usable digital format.

Tifinagh is used by millions of speakers across North Africa, yet computational support for this script remains limited. This project aims to bridge that gap by providing an accessible, accurate OCR solution for Tifinagh text recognition.

## ✨ Features

- **Character Recognition**: Accurately recognizes Tifinagh characters from images
- **CNN Architecture**: Leverages deep convolutional neural networks for robust feature extraction
- **Multiple Input Formats**: Supports various image formats (PNG, JPG, JPEG)
- **Preprocessing Pipeline**: Includes image enhancement and normalization techniques
- **Web Interface**: Interactive UI for easy text extraction (HTML/CSS)
- **Batch Processing**: Process multiple images efficiently
- **High Accuracy**: Achieves strong performance on both printed and handwritten text

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
TensorFlow/Keras
NumPy
OpenCV
Pillow
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/zacharyb02/OCR-tifinagh-script.git
cd OCR-tifinagh-script
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the pre-trained model** (if available)
```bash
# Place your trained model in the models/ directory
```

## 💻 Usage

### Web Interface

```bash
# Launch the web interface
python src/app.py
```

Navigate to `http://localhost:5000` in your browser to use the interactive OCR interface.

### Jupyter Notebook

Explore the training and inference notebooks:

```bash
jupyter notebook notebooks/
```

## 🏗️ Model Architecture

The OCR system employs a Convolutional Neural Network (CNN) architecture optimized for Tifinagh character recognition:

- **Input Layer**: Preprocessed grayscale images (normalized)
- **Convolutional Layers**: Multiple conv layers with ReLU activation for feature extraction
- **Pooling Layers**: Max pooling for spatial dimension reduction
- **Fully Connected Layers**: Dense layers for classification
- **Output Layer**: Softmax activation for character prediction

### Training Process

1. **Data Augmentation**: Rotation, scaling, and noise injection for robustness
2. **Preprocessing**: Grayscale conversion, normalization, and resizing
3. **Training**: Adam optimizer with categorical cross-entropy loss
4. **Validation**: K-fold cross-validation for model evaluation

## 📊 Dataset

The model is trained on a custom curated dataset containing:

- Tifinagh characters
- Printed Tifinagh text samples
- Various writing styles and sizes
- Augmented samples for improved generalization

### Character Set

The model recognizes the 33 characters of the standard Tifinagh alphabet:
```
ⴰ ⴱ ⴳ ⴷ ⴹ ⴻ ⴼ ⴽ ⵀ ⵃ ⵄ ⵅ ⵇ ⵉ ⵊ ⵍ ⵎ ⵏ ⵓ ⵔ ⵕ ⵖ ⵙ ⵚ ⵛ ⵜ ⵟ ⵡ ⵢ ⵣ ⵥ ⵧ ⵯ
```

## 🗂️ Project Structure

```
OCR-tifinagh-script/
├── images/               # Sample images and results
├── src/                  # Source code
│   ├── deployement/
│   │   ├── extracted_words/
│   │   ├── static/
│   │   ├── templates/
│   │   ├──deploy.py      # Web application
│   │   └── functions.py
│   ├── notebooks         # Jupyter notebooks for experiments
│   │   ├── segmentation.ipynb
│   │   ├── test.ipynb
│   │   ├── train.ipynb
│   │   └── deptest.ipynb
├── notebooks/           
├── models/              # Trained model weights
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Resources

- [Tifinagh Unicode Block](https://unicode.org/charts/PDF/U2D30.pdf)
- [Amazigh Language Resources](https://en.wikipedia.org/wiki/Tifinagh)
- [VGG16 Architecture Paper](https://arxiv.org/abs/1409.1556)

## 📧 Contact

**Zakaria Baou**
- GitHub: [@zacharyb02](https://github.com/zacharyb02)
**Mohammed Aoukicha**
- GitHub: [@mohaaoukicha-lgtm](https://github.com/mohaaoukicha-lgtm)
**Mohammed Taleb**
- GitHub: [@MOHAMMEDTALEB20](https://github.com/MOHAMMEDTALEB20)
