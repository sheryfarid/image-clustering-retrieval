"""
Complete Image Retrieval & Classification System
================================================
Works with your directory structure:
    D:\image-clustering-retrieval-20260104T172126Z-1-001\
    ├── app.py (this file)
    └── image-clustering-retrieval\
        ├── features\
        ├── models\
        ├── configs\
        └── dataset\ or object\

Run: streamlit run app.py
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import faiss
from PIL import Image
import os
import json
import pickle
from pathlib import Path
import glob

# ============================================
# IMPORTANT: Configure paths for YOUR structure
# ============================================

# Get the directory where app.py is located
APP_DIR = Path(__file__).parent

# Your models are in: D:\...\image-clustering-retrieval\
PROJECT_DIR = APP_DIR / "image-clustering-retrieval"

# Now set all paths relative to PROJECT_DIR
# DATASET_DIR = PROJECT_DIR / "dataset"  # or "object" if that's your folder name
FAISS_DIR = PROJECT_DIR / "models" / "faiss_index"
CLUSTERING_DIR = PROJECT_DIR / "models" / "clustering"
CLASSIFIER_DIR = PROJECT_DIR / "models" / "classifier"
FEATURES_DIR = PROJECT_DIR / "features"
CONFIG_DIR = PROJECT_DIR / "configs"

# Check if dataset is named "object" instead
DATASET_DIR = Path("D:/object")

# Verify it exists
if not DATASET_DIR.exists():
    print(f"Warning: Dataset not found at {DATASET_DIR}")
    DATASET_DIR = None
else:
    print(f"✅ Found dataset at: {DATASET_DIR}")

# ============================================
# Feature Extractor (ResNet-50)
# ============================================

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Load ResNet-50
        print("Loading ResNet-50...")
        model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        print("ResNet-50 loaded!")
    
    def extract(self, image):
        """Extract feature from PIL Image"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(img_tensor)
            feature = feature.view(feature.size(0), -1).cpu().numpy()
            # L2 normalize
            feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        return feature

# ============================================
# FAISS Retrieval System
# ============================================

class ImageRetrieval:
    def __init__(self):
        self.index = None
        self.image_paths = None
        self.labels = None
        self.categories = None
        
    def load_index(self, faiss_dir, dataset_dir):
        """Load FAISS index and fix image paths for local system"""
        print(f"Loading FAISS index from: {faiss_dir}")
        
        # Load FAISS index
        index_path = faiss_dir / "image_index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"FAISS index loaded: {self.index.ntotal} vectors")
        
        # Load paths and labels
        old_paths = np.load(faiss_dir / "index_paths.npy", allow_pickle=True)
        self.labels = np.load(faiss_dir / "index_labels.npy")
        
        # Convert old paths to local paths
        self.image_paths = []
        conversion_success = 0
        conversion_failed = 0
        
        for old_path in old_paths:
            old_path_str = str(old_path)
            parts = Path(old_path_str).parts
            
            try:
                # Remove any 'dataset' or 'object' folder from old path
                parts = [p for p in parts if p.lower() not in ('dataset', 'object')]
                
                # Take last two parts: category + filename
                if len(parts) >= 2:
                    relative = Path(parts[-2]) / parts[-1]
                else:
                    relative = Path(parts[-1])
                
                # Create new local path
                new_path = dataset_dir / relative
                self.image_paths.append(str(new_path))
                
                if new_path.exists():
                    conversion_success += 1
                else:
                    conversion_failed += 1
                    
            except Exception as e:
                # Fallback: just use filename in dataset root
                filename = Path(old_path_str).name
                new_path = dataset_dir / filename
                self.image_paths.append(str(new_path))
                conversion_failed += 1
        
        self.image_paths = np.array(self.image_paths)
        
        print(f"Path conversion: {conversion_success} found, {conversion_failed} missing")
        if conversion_success > 0:
            sample_idx = np.where([Path(p).exists() for p in self.image_paths])[0][0]
            print(f"Sample converted path: {self.image_paths[sample_idx]}")
        if conversion_failed > 0:
            sample_idx = np.where([not Path(p).exists() for p in self.image_paths])[0][0]
            print(f"Sample failed path: {self.image_paths[sample_idx]}")
            print(f"Looking in dataset: {dataset_dir}")

        
    def load_metadata(self, config_dir):
        """Load dataset metadata"""
        metadata_path = config_dir / "dataset_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.categories = metadata.get('categories', [])
                print(f"Loaded {len(self.categories)} categories from metadata")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.categories = []
        
        # Fallback: extract from dataset directory
        if not self.categories and DATASET_DIR.exists():
            self.categories = sorted([
                d.name for d in DATASET_DIR.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
            print(f"Auto-detected {len(self.categories)} categories from dataset folder")
    
    def search(self, query_feature, k=10):
        """Search top-K similar images"""
        query_feature = query_feature.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_feature, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths):
                similarity = 1 / (1 + dist)
                results.append({
                    "index": int(idx),
                    "image_path": self.image_paths[idx],
                    "label": int(self.labels[idx]),
                    "category": self.categories[self.labels[idx]] if self.categories and self.labels[idx] < len(self.categories) else f"Category_{self.labels[idx]}",
                    "distance": float(dist),
                    "similarity": float(similarity)
                })
        
        return results

# ============================================
# Clustering System
# ============================================

class ClusteringSystem:
    def __init__(self):
        self.kmeans = None
        self.cluster_labels = None
        self.metrics = None
        
    def load_clustering(self, clustering_dir):
        """Load K-Means clustering model"""
        if not clustering_dir.exists():
            return False
        
        # Find kmeans pickle file
        kmeans_files = list(clustering_dir.glob("kmeans_k*.pkl"))
        if not kmeans_files:
            return False
        
        try:
            # Load K-Means model
            with open(kmeans_files[0], 'rb') as f:
                self.kmeans = pickle.load(f)
            
            # Load cluster labels
            labels_path = clustering_dir / "cluster_labels.npy"
            if labels_path.exists():
                self.cluster_labels = np.load(labels_path)
            
            # Load metrics
            metrics_path = clustering_dir / "clustering_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, encoding='utf-8') as f:
                    self.metrics = json.load(f)
            
            print(f"Clustering model loaded: {self.metrics.get('n_clusters', 'N/A')} clusters")
            return True
        except Exception as e:
            print(f"Error loading clustering: {e}")
            return False
    
    def predict_cluster(self, feature):
        """Predict cluster for a feature"""
        if self.kmeans is None:
            return None
        try:
            return self.kmeans.predict(feature.reshape(1, -1))[0]
        except:
            return None

# ============================================
# Classification System
# ============================================

class ClassificationSystem:
    def __init__(self):
        self.classifier = None
        self.pca = None
        self.scaler = None
        self.results = None
        
    def load_classifier(self, classifier_dir):
        """Load Random Forest classifier and preprocessing"""
        if not classifier_dir.exists():
            return False
        
        try:
            # Load PCA (optional)
            pca_path = classifier_dir / "pca.pkl"
            if pca_path.exists():
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
            
            # Load Scaler
            scaler_path = classifier_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load Random Forest
            rf_path = classifier_dir / "random_forest.pkl"
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.classifier = pickle.load(f)
            
            # Load results
            results_path = classifier_dir / "classification_results.json"
            if results_path.exists():
                with open(results_path, encoding='utf-8') as f:
                    self.results = json.load(f)
            
            print("Classification model loaded!")
            return True
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return False
    
    def predict(self, feature):
        """Predict category for a feature"""
        if self.classifier is None:
            return None, None
        
        try:
            # Apply PCA if available
            if self.pca is not None:
                feature = self.pca.transform(feature.reshape(1, -1))
            else:
                feature = feature.reshape(1, -1)
            
            # Apply scaling
            if self.scaler is not None:
                feature = self.scaler.transform(feature)
            
            # Predict
            pred_label = self.classifier.predict(feature)[0]
            pred_proba = self.classifier.predict_proba(feature)[0]
            confidence = pred_proba[pred_label]
            
            return pred_label, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None

# ============================================
# Streamlit App
# ============================================

@st.cache_resource
def load_all_models():
    """Load all models once and cache"""
    models_status = {}
    errors = []
    
    # 1. Feature Extractor (Always needed)
    try:
        extractor = FeatureExtractor(device='cpu')
        models_status['feature_extractor'] = True
    except Exception as e:
        extractor = None
        models_status['feature_extractor'] = False
        errors.append(f"Feature Extractor: {e}")
    
    # 2. FAISS Retrieval
    retrieval = ImageRetrieval()
    try:
        retrieval.load_index(FAISS_DIR, DATASET_DIR)
        retrieval.load_metadata(CONFIG_DIR)
        models_status['retrieval'] = True
    except Exception as e:
        errors.append(f"FAISS Retrieval: {e}")
        models_status['retrieval'] = False
        retrieval = None
    
    # 3. Clustering (Optional)
    clustering = ClusteringSystem()
    try:
        if clustering.load_clustering(CLUSTERING_DIR):
            models_status['clustering'] = True
        else:
            models_status['clustering'] = False
            clustering = None
    except Exception as e:
        errors.append(f"Clustering: {e}")
        models_status['clustering'] = False
        clustering = None
    
    # 4. Classification (Optional)
    classifier = ClassificationSystem()
    try:
        if classifier.load_classifier(CLASSIFIER_DIR):
            models_status['classification'] = True
        else:
            models_status['classification'] = False
            classifier = None
    except Exception as e:
        errors.append(f"Classification: {e}")
        models_status['classification'] = False
        classifier = None
    
    return extractor, retrieval, clustering, classifier, models_status, errors

def main():
    st.set_page_config(
        page_title="Image Retrieval System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Image Retrieval & Clustering System")
    st.markdown("Upload an image to find similar images and get predictions")
    
    # Show directory info
    with st.expander("📁 Directory Configuration & Debug Info", expanded=False):
        st.code(f"""
App Directory:      {APP_DIR}
Project Directory:  {PROJECT_DIR}
Dataset:            {DATASET_DIR}
Dataset Exists:     {DATASET_DIR.exists()}
FAISS Index:        {FAISS_DIR}
Clustering:         {CLUSTERING_DIR}
Classifier:         {CLASSIFIER_DIR}
        """)
        
        # Show dataset structure
        if DATASET_DIR.exists():
            categories = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()][:10]
            st.write("Sample categories found:")
            st.write(categories)
            
            # Check one category
            if categories:
                sample_cat = DATASET_DIR / categories[0]
                sample_images = list(sample_cat.glob("*.*"))[:5]
                st.write(f"\nSample images in '{categories[0]}':")
                for img in sample_images:
                    st.write(f"  - {img.name}")
        else:
            st.error(f"Dataset directory does not exist: {DATASET_DIR}")
    
    # Load models
    with st.spinner("🔄 Loading models... (first run may take 30-60 seconds)"):
        extractor, retrieval, clustering, classifier, models_status, errors = load_all_models()
    
    # Show any errors
    if errors:
        with st.expander("⚠️ Loading Warnings", expanded=False):
            for error in errors:
                st.warning(error)
    
    # Sidebar - System Status
    with st.sidebar:
        st.header("🎯 System Status")
        
        for model_name, status in models_status.items():
            if status:
                st.success(f"✅ {model_name.replace('_', ' ').title()}")
            else:
                st.error(f"❌ {model_name.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        k = st.slider("Number of results", min_value=5, max_value=20, value=9)
        show_predictions = st.checkbox("Show Classification", value=True)
        show_cluster = st.checkbox("Show Cluster Info", value=True)
        
        st.markdown("---")
        
        # Model Info
        if retrieval and models_status['retrieval']:
            st.info(f"""
            **FAISS Index:**
            - Images: {retrieval.index.ntotal:,}
            - Categories: {len(retrieval.categories)}
            """)
        
        if clustering and models_status['clustering'] and clustering.metrics:
            st.info(f"""
            **Clustering:**
            - Clusters: {clustering.metrics.get('n_clusters', 'N/A')}
            - Silhouette: {clustering.metrics.get('silhouette_score', 0):.3f}
            """)
        
        if classifier and models_status['classification'] and classifier.results:
            st.info(f"""
            **Classification:**
            - Accuracy: {classifier.results.get('validation_accuracy', 0):.1%}
            """)
    
    # Main content
    if not models_status['retrieval']:
        st.error("❌ FAISS index not loaded. Cannot perform retrieval.")
        st.info("Please check that these files exist:")
        st.code(f"""
{FAISS_DIR}/image_index.faiss
{FAISS_DIR}/index_paths.npy
{FAISS_DIR}/index_labels.npy
{DATASET_DIR}/  (with category folders)
        """)
        st.stop()
    
    st.success(f"✅ System ready! Indexed: {retrieval.index.ntotal:,} images from {len(retrieval.categories)} categories")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to find similar images"
    )
    
    if uploaded_file is not None:
        # Display query image
        query_image = Image.open(uploaded_file).convert("RGB")
        
        # Layout
        col_query, col_info = st.columns([1, 2])
        
        with col_query:
            st.subheader("📤 Query Image")
            st.image(query_image, width=300)
        
        with col_info:
            st.subheader("📊 Analysis")
            
            # Extract features
            with st.spinner("🔍 Extracting features..."):
                query_feature = extractor.extract(query_image)
            
            st.success("✅ Features extracted!")
            
            # Classification
            if show_predictions and classifier and models_status['classification']:
                pred_label, confidence = classifier.predict(query_feature)
                if pred_label is not None and pred_label < len(retrieval.categories):
                    pred_category = retrieval.categories[pred_label]
                    
                    st.markdown("### 🎯 Classification")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category", pred_category)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
            
            # Clustering
            if show_cluster and clustering and models_status['clustering']:
                cluster_id = clustering.predict_cluster(query_feature)
                if cluster_id is not None:
                    st.markdown("### 🎨 Cluster")
                    st.info(f"Cluster **{cluster_id}**")
        
        # Retrieval Results
        st.markdown("---")
        st.subheader(f"🔎 Top {k} Similar Images")
        
        with st.spinner("🔍 Searching..."):
            results = retrieval.search(query_feature, k=k)
        
        if not results:
            st.warning("No results found.")
        else:
            # Display grid
            cols = st.columns(5)
            
            found_count = 0
            missing_count = 0
            
            for i, result in enumerate(results):
                with cols[i % 5]:
                    img_path = result['image_path']
                    
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            st.image(img, width=200)
                            found_count += 1
                            
                            st.markdown(f"""
                            **{result['category']}**  
                            Similarity: `{result['similarity']:.3f}`
                            """)
                            
                            if show_cluster and clustering and models_status['clustering']:
                                img_idx = result['index']
                                if img_idx < len(clustering.cluster_labels):
                                    st.caption(f"Cluster: {clustering.cluster_labels[img_idx]}")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                            missing_count += 1
                    else:
                        st.warning(f"⚠️ Not found")
                        st.caption(f"{result['category']}")
                        st.caption(f"`{Path(img_path).name}`")
                        missing_count += 1
                        
                        # Show the full path being tried (for debugging)
                        with st.expander("🔍 Debug", expanded=False):
                            st.text(f"Looking for:\n{img_path}")
                            st.text(f"\nDataset dir:\n{DATASET_DIR}")
            
            # Show summary
            if missing_count > 0:
                st.warning(f"⚠️ {missing_count} out of {len(results)} images not found on disk")
                st.info("💡 Tip: Expand '🔍 Debug' under any missing image to see the exact path being searched")
            
            # Stats
            st.markdown("---")
            st.subheader("📈 Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                categories_found = [r['category'] for r in results]
                unique_cats = len(set(categories_found))
                st.metric("Unique Categories", unique_cats)
            
            with col2:
                avg_sim = np.mean([r['similarity'] for r in results])
                st.metric("Avg Similarity", f"{avg_sim:.3f}")
            
            with col3:
                top_cat = max(set(categories_found), key=categories_found.count)
                st.metric("Most Frequent", top_cat)

if __name__ == "__main__":
    # Validate paths
    if not PROJECT_DIR.exists():
        st.error(f"""
        ❌ Project directory not found!
        
        Looking for: {PROJECT_DIR}
        
        Your structure should be:
        {APP_DIR}/
        └── image-clustering-retrieval/
            ├── features/
            ├── models/
            ├── configs/
            └── dataset/ (or object/)
        """)
        st.stop()
    
    main()