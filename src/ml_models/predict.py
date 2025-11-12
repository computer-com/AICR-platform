"""
Step 3D: Prediction/Inference Script
Load trained models and make predictions on new data
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Download NLTK data if needed
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

MODEL_DIR = 'src/ml_models'

class CrisisPredictor:
    """Wrapper class for making predictions using trained models"""
    
    def __init__(self, model_type='text'):
        """
        Initialize predictor
        
        Args:
            model_type: 'text', 'multimodal', or 'image'
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.geo_scaler = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model and associated artifacts"""
        print(f"Loading {self.model_type} model...")
        
        try:
            if self.model_type == 'text':
                model_path = os.path.join(MODEL_DIR, 'text_model.h5')
                tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
                le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
                config_path = os.path.join(MODEL_DIR, 'model_config.pkl')
            elif self.model_type == 'multimodal':
                model_path = os.path.join(MODEL_DIR, 'multimodal_model.h5')
                tokenizer_path = os.path.join(MODEL_DIR, 'multimodal_tokenizer.pkl')
                le_path = os.path.join(MODEL_DIR, 'multimodal_label_encoder.pkl')
                config_path = os.path.join(MODEL_DIR, 'multimodal_config.pkl')
            elif self.model_type == 'image':
                model_path = os.path.join(MODEL_DIR, 'image_model.h5')
                config_path = os.path.join(MODEL_DIR, 'image_model_config.pkl')
                le_path = os.path.join(MODEL_DIR, 'image_class_indices.pkl')
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load model
            self.model = load_model(model_path)
            print(f"[OK] Model loaded from {model_path}")
            
            # Load config
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"[OK] Config loaded")
            
            # Load tokenizer (for text models)
            if self.model_type in ['text', 'multimodal']:
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                with open(le_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"[OK] Tokenizer and label encoder loaded")
            
            # Load geospatial scaler (for multimodal)
            if self.model_type == 'multimodal' and self.config.get('has_geo', False):
                scaler_path = os.path.join(MODEL_DIR, 'geo_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.geo_scaler = pickle.load(f)
                    print(f"[OK] Geospatial scaler loaded")
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ''
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        try:
            tokens = nltk.word_tokenize(text)
        except:
            tokens = text.split()
        # Remove stopwords
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return ' '.join(tokens)
    
    def predict_text(self, text, latitude=None, longitude=None):
        """
        Predict crisis type from text (and optionally geospatial data)
        
        Args:
            text: Description text
            latitude: Optional latitude
            longitude: Optional longitude
            
        Returns:
            dict with prediction results
        """
        if self.model_type not in ['text', 'multimodal']:
            raise ValueError(f"Model type {self.model_type} doesn't support text prediction")
        
        # Clean text
        clean = self.clean_text(text)
        
        # Tokenize and pad
        seq = self.tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=self.config['max_len'], padding='post')
        
        # Make prediction
        if self.model_type == 'multimodal' and self.config.get('has_geo', False):
            if latitude is None or longitude is None:
                raise ValueError("Multimodal model requires latitude and longitude")
            
            # Prepare geospatial features
            geo_features = np.array([[latitude, longitude]])
            if self.geo_scaler:
                geo_features = self.geo_scaler.transform(geo_features)
            
            # Predict with both inputs
            predictions = self.model.predict([padded, geo_features], verbose=0)[0]
        else:
            # Predict with text only
            predictions = self.model.predict(padded, verbose=0)[0]
        
        # Get top prediction
        top_idx = predictions.argmax()
        top_class = self.label_encoder.classes_[top_idx]
        top_confidence = float(predictions[top_idx])
        
        # Get all predictions sorted by confidence
        all_predictions = [
            {
                'crisis_type': self.label_encoder.classes_[i],
                'confidence': float(predictions[i])
            }
            for i in np.argsort(predictions)[::-1]
        ]
        
        return {
            'predicted_class': top_class,
            'confidence': top_confidence,
            'all_predictions': all_predictions,
            'raw_probabilities': predictions.tolist()
        }
    
    def predict_batch(self, texts, latitudes=None, longitudes=None):
        """Predict for multiple texts at once"""
        results = []
        
        for i, text in enumerate(texts):
            lat = latitudes[i] if latitudes else None
            lon = longitudes[i] if longitudes else None
            
            result = self.predict_text(text, lat, lon)
            results.append(result)
        
        return results


def demo_predictions():
    """Run demo predictions with sample data"""
    
    print("\n" + "="*60)
    print("Crisis Prediction Demo")
    print("="*60)
    
    # Sample crisis reports
    sample_reports = [
        {
            'text': 'Water is rising rapidly on Main Street. Roads are completely flooded.',
            'lat': 43.6532,
            'lon': -79.3832
        },
        {
            'text': 'Fire broke out in building. Smoke visible from multiple blocks away.',
            'lat': 43.6510,
            'lon': -79.3470
        },
        {
            'text': 'Strong earthquake felt. Buildings shaking. People evacuating.',
            'lat': 43.6485,
            'lon': -79.3850
        },
        {
            'text': 'Fallen trees blocking the highway after severe storm.',
            'lat': 43.6550,
            'lon': -79.3800
        }
    ]
    
    # Try to load the best available model
    for model_type in ['multimodal', 'text']:
        try:
            predictor = CrisisPredictor(model_type=model_type)
            print(f"\n[OK] Using {model_type} model\n")
            break
        except:
            continue
    else:
        print("ERROR: No trained model found!")
        print("Please run text_model.py or train_multimodal.py first")
        return
    
    # Make predictions
    print("\nSample Predictions:")
    print("-" * 60)
    
    for i, report in enumerate(sample_reports, 1):
        print(f"\n[Report {i}]")
        print(f"Text: {report['text']}")
        print(f"Location: ({report['lat']}, {report['lon']})")
        
        try:
            result = predictor.predict_text(
                report['text'],
                report['lat'],
                report['lon']
            )
            
            print(f"\nPrediction: {result['predicted_class'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            print("\nTop 3 predictions:")
            for j, pred in enumerate(result['all_predictions'][:3], 1):
                print(f"  {j}. {pred['crisis_type']}: {pred['confidence']:.2%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 60)


if __name__ == '__main__':
    # Run demo
    demo_predictions()
    
    print("\n" + "="*60)
    print("Prediction Script Complete")
    print("="*60)
    print("\nYou can now use this in your API:")
    print("  from predict import CrisisPredictor")
    print("  predictor = CrisisPredictor(model_type='multimodal')")
    print("  result = predictor.predict_text(text, lat, lon)")
    print("="*60)