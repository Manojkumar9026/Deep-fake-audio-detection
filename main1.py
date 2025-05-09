# Import necessary libraries
import os
import numpy as np
import librosa
import joblib
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI(title="Deep Fake Voice Detection API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# SET YOUR FOLDER PATHS HERE
# ==============================================

REAL_FOLDER_PATH = "D:\Deepfake_project\KAGGLE\AUDIO\FAKE"  
FAKE_FOLDER_PATH = "D:\Deepfake_project\KAGGLE\AUDIO\REAL"  
# ==============================================

# Paths to save the trained models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to extract audio features
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    """Extract features from audio file for classification"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Set duration to 3 seconds (trim or pad)
        if len(y) > sr * 3:
            y = y[:sr * 3]
        else:
            y = np.pad(y, (0, max(0, sr * 3 - len(y))), 'constant')
        
        features = []
        
        # Extract features
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            features.extend(mfccs)
            
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            features.extend(chroma)
            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            features.extend(mel)
            
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Function to process all audio files in a directory
def process_audio_directory(directory):
    features = []
    labels = []
    
    # Determine if this is a "REAL" or "FAKE" directory
    is_real = "REAL" in directory
    label = 1 if is_real else 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(root, file)
                
                # Extract features from the audio file
                audio_features = extract_features(file_path)
                
                if audio_features is not None:
                    features.append(audio_features)
                    labels.append(label)
    
    return features, labels

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

@app.post("/train")
async def train_model(real_folder: str = Form(None), fake_folder: str = Form(None)):
    """
    Train multiple models using Real and Fake audio folders and compare their performance.
    If no folders are provided, use the default paths set at the top of the file.
    """
    try:
        # Use provided folders or fall back to defaults
        real_dir = real_folder if real_folder else REAL_FOLDER_PATH
        fake_dir = fake_folder if fake_folder else FAKE_FOLDER_PATH
        
        # Validate directories
        if not os.path.isdir(real_dir):
            raise HTTPException(status_code=400, detail=f"Real folder path '{real_dir}' is not a valid directory")
        if not os.path.isdir(fake_dir):
            raise HTTPException(status_code=400, detail=f"Fake folder path '{fake_dir}' is not a valid directory")
        
        # Process real audio files
        print(f"Processing real audio files from: {real_dir}")
        real_features, real_labels = process_audio_directory(real_dir)
        
        # Process fake audio files
        print(f"Processing fake audio files from: {fake_dir}")
        fake_features, fake_labels = process_audio_directory(fake_dir)
        
        # Combine datasets
        X = np.array(real_features + fake_features)
        y = np.array(real_labels + fake_labels)
        
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid audio files found in the provided directories")
        
        print(f"Total samples: {len(X)}, Real: {sum(y)}, Fake: {len(y) - sum(y)}")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models to train
        models = {
            "xgboost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            "svm": SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        # Results dictionary
        results = {}
        
        # Train and evaluate each model
        best_model = None
        best_accuracy = 0.0
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True)
            
            # Save the model
            model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            
            # Store results
            results[model_name] = {
                "accuracy": accuracy,
                "classification_report": convert_numpy_types(report)
            }
            
            # Check if this is the best model so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        # Save the best model as the default model
        best_model_path = os.path.join(MODEL_DIR, f"{best_model}_model.pkl")
        default_model_path = os.path.join(MODEL_DIR, "default_model.pkl")
        # Copy the best model to the default model path
        best_model_obj = joblib.load(best_model_path)
        joblib.dump(best_model_obj, default_model_path)
        
        # Return comprehensive results
        return {
            "message": "Multiple models trained successfully",
            "total_samples": int(len(X)),
            "real_samples": int(sum(y)),
            "fake_samples": int(len(y) - sum(y)),
            "real_folder_used": real_dir,
            "fake_folder_used": fake_dir,
            "results": results,
            "best_model": {
                "name": best_model,
                "accuracy": float(best_accuracy)
            },
            "models_saved": list(models.keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...), model_name: str = Form("default")):
    """
    Predict if the uploaded audio file is real or fake using the specified model
    """
    try:
        # Determine which model to use
        if model_name == "default":
            model_path = os.path.join(MODEL_DIR, "default_model.pkl")
        else:
            model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        
        # Check if model exists
        if not os.path.exists(model_path):
            available_models = [f.replace("_model.pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith("_model.pkl")]
            if not available_models:
                raise HTTPException(status_code=400, detail="No models trained yet. Please train models first.")
            else:
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Available models: {', '.join(available_models)}")
        
        # Save the uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        # Extract features
        features = extract_features(temp_file)
        
        if features is None:
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail="Could not extract features from the uploaded file")
        
        # Load model and make prediction
        clf = joblib.load(model_path)
        features_array = np.array(features).reshape(1, -1)
        prediction = int(clf.predict(features_array)[0])
        probabilities = clf.predict_proba(features_array)[0]
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Return prediction results
        result_label = "REAL" if prediction == 1 else "FAKE"
        confidence = float(probabilities[prediction])
        
        return {
            "filename": file.filename,
            "model_used": model_name,
            "prediction": result_label,
            "confidence": confidence,
            "probability_fake": float(probabilities[0]),
            "probability_real": float(probabilities[1])
        }
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    print(f"Starting Deep Fake Voice Detection API with multiple ML classifiers")
    print(f"Default REAL folder path: {REAL_FOLDER_PATH}")
    print(f"Default FAKE folder path: {FAKE_FOLDER_PATH}")
    print(f"Models will be saved to: {MODEL_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)