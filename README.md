# 🎭 Face Symmetry Studio

[![CI](https://github.com/Sais-gallego/face-symmetry-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/face-symmetry-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)

**Face Symmetry Studio** is an open-source biometric facial symmetry analysis and mirror synthesis engine built with **OpenCV**, **NumPy**, and **Flask**. It analyzes facial asymmetry along the midline sagittal plane, generates seamless bilateral mirrored faces (Left-Left and Right-Right symmetric clones), computes Golden Ratio ($\Phi$) adherence, and visualizes anatomical divergences using thermal heatmaps.

---

## 🌟 Key Features

- **Midline Sagittal Axis Detection**: Automatically locates the facial midline passing through pupil midpoints and the nasal bridge.
- **Bilateral Face Cloning & Seamless Mirroring**:
  - **Left-Left Symmetrical Clone**: Mirrors the left facial hemisphere onto the right.
  - **Right-Right Symmetrical Clone**: Mirrors the right facial hemisphere onto the left.
  - **Cosine Feathering & Seam Blending**: Eliminates harsh central creases for natural morphological synthesis.
- **Asymmetry Thermal Heatmap**: High-contrast `JET` color mapping highlighting subtle anatomical divergences in eyebrows, cheekbones, and jawline.
- **Biometric Symmetry Scoring Index**:
  - *Pixel Similarity Score*: Structural pixel-level correlation across mirrored hemispheres.
  - *Ocular Balance*: Inter-pupillary distance equilibrium relative to the midline.
  - *Golden Ratio ($\Phi \approx 1.618$)*: Vertical-to-horizontal facial proportion analysis.
- **Dual Interface**:
  - **CLI**: Fast batch analysis and high-resolution 4-panel diagnostic exports.
  - **Interactive Web Studio**: Built-in dark-mode Flask web dashboard with live demo and drag-and-drop image uploads.

---

## 📁 Diagnostic Output Architecture

```
results/
├── {name}_symmetry_analysis.png   # 4-Panel diagnostic collage (Original, Left-Sym, Heatmap, Right-Sym)
├── {name}_left_symmetric.png      # High-res Left-Left mirrored clone
├── {name}_right_symmetric.png     # High-res Right-Right mirrored clone
└── {name}_asymmetry_heatmap.png   # Anatomical divergence thermal heatmap
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Sais-gallego/face-symmetry-studio.git
cd face-symmetry-studio
pip install -r requirements.txt
```

### 2. Command Line Usage

Run demonstration on synthetic portrait:
```bash
python -m face_symmetry.cli --demo
```

Analyze your own portrait:
```bash
python -m face_symmetry.cli --image path/to/portrait.jpg --output-dir results/
```

### 3. Launch Interactive Web Studio

```bash
python -m face_symmetry.cli --web --port 5050
```
Open [http://localhost:5050](http://localhost:5050) in your browser.

---

## 🧪 Testing

Run automated unit test suite:

```bash
python -m unittest discover tests -v
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
