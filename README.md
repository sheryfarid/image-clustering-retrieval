# Image Clustering, Classification & Retrieval System

## 📌 Overview
This project implements an **end-to-end image understanding system** that performs **feature extraction, clustering, classification, and image retrieval** using deep learning and classical machine learning techniques.  
It is built on the **Caltech-101 dataset** and uses a **pretrained ResNet-50** model for robust visual feature embeddings.

---

## 🎯 Objectives
- Extract discriminative deep features from images using transfer learning
- Compare **unsupervised (K-Means, Hierarchical)** and **supervised (SVM, Random Forest)** models
- Build an **interactive image retrieval system** with measurable performance metrics

---

## 🧠 System Architecture (High Level)
1. Load images from Caltech-101  
2. Preprocess images (resize, normalize)  
3. Extract 2048-D features using **ResNet-50**  
4. Normalize features (L2)  
5. Apply clustering (K-Means & Hierarchical)  
6. Train classifiers (SVM & Random Forest)  
7. Build FAISS index for fast retrieval  
8. Serve results via **Streamlit web app**

---

## 🔍 Models & Techniques Used

### Feature Extraction
- **ResNet-50 (pretrained, frozen)**  
- Output: 2048-dimensional embeddings

### Clustering
- **K-Means** (baseline clustering)  
- **Hierarchical Agglomerative Clustering (K=50)**  
- Metrics: Silhouette Score, Davies–Bouldin Index

### Classification
- **Support Vector Machine (RBF kernel)** → Best performance (~86%)  
- **Random Forest** → Baseline comparison

### Retrieval
- **FAISS (Approximate Nearest Neighbors)**  
- Metric: Precision@10 ≈ 0.98

---

## 🌐 Streamlit Application
The Streamlit app allows users to:
- Upload an image from the dataset  
- Retrieve visually similar images in real time  
- View similarity probabilities and image names  
- Interact with the trained models without retraining

---

## 📊 Results Summary
- **SVM Validation Accuracy:** ~85–86%  
- **Clustering Quality:** Moderate (overlapping classes)  
- **Retrieval Performance:** Very strong (near-perfect Precision@10)  

> **Key insight:**  
> Even when clustering quality is weak, deep embeddings still enable excellent image retrieval.

---

## 📁 Project Structure
```

image-clustering-retrieval/
├── configs/
├── data/
├── features/
├── models/
│   ├── classifier/
│   ├── clustering/
│   └── faiss_index/
├── results/
├── app.py
├── final_report.json
└── README.md

````

---

## ⚙️ Setup & Run
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
````

---

## 🚀 Future Improvements

* Replace ResNet-50 with **ViT or EfficientNet**
* Improve clustering using **deep or spectral clustering**
* Scale retrieval to larger datasets
* Add fairness and bias evaluation per category

---

## 📌 Conclusion

This project demonstrates a **practical, real-world image understanding pipeline** combining deep learning with classical ML.
While clustering remains challenging, **classification and retrieval results are strong**, proving the effectiveness of deep feature embeddings.

---

## 👥 Team Members

* Abdul Basit (22F-BSAI-25)
* Shehryar Ahmed (22F-BSAI-28)
* Muhammad Hamamd (22F-BSAI-39)
* Muhammad Sohaib (22F-BSAI-40)

```

