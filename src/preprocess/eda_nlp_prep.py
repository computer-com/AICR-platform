"""
Step 2: EDA and NLP Preprocessing Script
Run with: python src/preprocess/eda_nlp_prep.py
"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# ---- ensure NLTK downloads only once ----
print("Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

from nltk.corpus import stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    print("Warning: Could not load stopwords, using empty set")
    stop_words = set()

# ---- load dataset ----
data_path = os.path.join('data', 'mock_crisis_reports.csv')

# Check if file exists
if not os.path.exists(data_path):
    print(f"ERROR: File not found: {data_path}")
    print("Creating sample mock_crisis_reports.csv file...")
    
    # Create sample data
    sample_data = {
        'id': range(1, 21),
        'description': [
            'Severe flooding on Main Street, water rising rapidly',
            'Fire reported in downtown building, multiple floors affected',
            'Earthquake felt across the city, buildings shaking',
            'Flooded basement, need immediate help',
            'Smoke visible from residential area',
            'Roads completely submerged in water',
            'Gas leak detected near school',
            'Multiple trees down blocking highway',
            'Power outage affecting entire neighborhood',
            'Building collapse after tremor',
            'Flash flood warning, evacuation needed',
            'Fire spreading to adjacent properties',
            'Aftershock felt, people panicking',
            'Water contamination reported',
            'Wildfire approaching residential area',
            'Major flooding at intersection',
            'Structural damage to bridge',
            'Gas explosion in commercial district',
            'Mudslide blocking main road',
            'Emergency shelter needed for displaced families'
        ],
        'crisis_type': [
            'flood', 'fire', 'earthquake', 'flood', 'fire',
            'flood', 'gas_leak', 'storm', 'power_outage', 'earthquake',
            'flood', 'fire', 'earthquake', 'flood', 'fire',
            'flood', 'earthquake', 'gas_leak', 'landslide', 'flood'
        ],
        'latitude': [
            43.6532, 43.6510, 43.6485, 43.6550, 43.6520,
            43.6505, 43.6600, 43.6530, 43.6540, 43.6490,
            43.6560, 43.6515, 43.6495, 43.6535, 43.6525,
            43.6545, 43.6555, 43.6500, 43.6510, 43.6565
        ],
        'longitude': [
            -79.3832, -79.3470, -79.3850, -79.3800, -79.3900,
            -79.3995, -79.3700, -79.3820, -79.3880, -79.3860,
            -79.3750, -79.3920, -79.3840, -79.3810, -79.3890,
            -79.3870, -79.3790, -79.3950, -79.3780, -79.3720
        ],
        'timestamp': [
            '2025-11-10T14:12:00Z', '2025-11-10T15:30:00Z', '2025-11-10T16:45:00Z',
            '2025-11-11T08:20:00Z', '2025-11-11T09:15:00Z', '2025-11-11T10:30:00Z',
            '2025-11-11T11:45:00Z', '2025-11-11T13:00:00Z', '2025-11-11T14:30:00Z',
            '2025-11-11T16:00:00Z', '2025-11-12T07:15:00Z', '2025-11-12T08:30:00Z',
            '2025-11-12T09:45:00Z', '2025-11-12T11:00:00Z', '2025-11-12T12:15:00Z',
            '2025-11-12T13:30:00Z', '2025-11-12T14:45:00Z', '2025-11-12T16:00:00Z',
            '2025-11-12T17:15:00Z', '2025-11-12T18:30:00Z'
        ],
        'source': ['citizen'] * 20
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(data_path, index=False)
    print(f"Created sample file: {data_path}")

df = pd.read_csv(data_path)

print(f"\nLoaded {len(df)} records")
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# ---- quick visualization ----
print("\nCreating crisis type distribution plot...")
plt.figure(figsize=(10, 6))
sns.countplot(y='crisis_type', data=df, order=df['crisis_type'].value_counts().index)
plt.title('Crisis Type Distribution')
plt.xlabel('Count')
plt.ylabel('Crisis Type')
plt.tight_layout()
os.makedirs('data', exist_ok=True)
plt.savefig('data/eda_crisis_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: data/eda_crisis_distribution.png")

# ---- text cleaning ----
print("\nCleaning text data...")

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except:
        # Fallback to simple split if nltk fails
        tokens = text.split()
    # Remove stopwords and short words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['description'].apply(clean_text)
print(f"[OK] Cleaned {len(df)} text entries")

# ---- word cloud ----
print("\nGenerating word cloud...")
text = ' '.join(df['clean_text'])
if text.strip():
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Crisis Reports', fontsize=16)
    plt.tight_layout()
    plt.savefig('data/eda_wordcloud.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: data/eda_wordcloud.png")
else:
    print("[WARNING] No text available for word cloud")

# ---- TF-IDF embedding ----
print("\nGenerating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=500, min_df=1)
X_text = vectorizer.fit_transform(df['clean_text'])
print(f'[OK] TF-IDF matrix shape: {X_text.shape}')
print(f"[OK] Number of features: {len(vectorizer.get_feature_names_out())}")

# ---- optional geospatial preview ----
if {'latitude', 'longitude'}.issubset(df.columns):
    print("\nCreating geospatial map...")
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # Try to plot with world map
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            fig, ax = plt.subplots(figsize=(12, 8))
            world.plot(ax=ax, color='lightgray', edgecolor='white')
            gdf.plot(ax=ax, color='red', markersize=50, alpha=0.6, label='Crisis Reports')
            plt.title('Crisis Reports Geographic Distribution', fontsize=14)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/eda_geospatial_map.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("[OK] Saved: data/eda_geospatial_map.png")
        except Exception as e:
            # Simple scatter plot as fallback
            print(f"[INFO] Using simple scatter plot: {e}")
            plt.figure(figsize=(10, 8))
            plt.scatter(df['longitude'], df['latitude'], c='red', s=100, alpha=0.6)
            plt.title('Crisis Reports Geographic Distribution')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/eda_geospatial_map.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("[OK] Saved: data/eda_geospatial_map.png (simple version)")
            
    except ImportError:
        print("[INFO] GeoPandas not available, creating simple scatter plot...")
        plt.figure(figsize=(10, 8))
        plt.scatter(df['longitude'], df['latitude'], c='red', s=100, alpha=0.6)
        plt.title('Crisis Reports Geographic Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/eda_geospatial_map.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Saved: data/eda_geospatial_map.png")
    except Exception as e:
        print(f"[WARNING] Could not create geospatial map: {e}")

# ---- save cleaned data ----
print("\nSaving processed data...")
os.makedirs('data/processed', exist_ok=True)
out_path = os.path.join('data', 'processed', 'clean_crisis_reports.csv')
df.to_csv(out_path, index=False)
print(f"[OK] Cleaned dataset saved to {out_path}")

# ---- Summary statistics ----
print("\n=== Summary Statistics ===")
print(f"Total records: {len(df)}")
print(f"Crisis types: {df['crisis_type'].nunique()}")
print(f"\nCrisis type counts:")
print(df['crisis_type'].value_counts())
print(f"\nAverage text length (original): {df['description'].str.len().mean():.1f} characters")
print(f"Average text length (cleaned): {df['clean_text'].str.len().mean():.1f} characters")

print("\n" + "="*50)
print("Step 2 complete - EDA and NLP Preprocessing [OK]")
print("="*50)
print("\nGenerated files:")
print("  - data/eda_crisis_distribution.png")
print("  - data/eda_wordcloud.png")
print("  - data/eda_geospatial_map.png")
print("  - data/processed/clean_crisis_reports.csv")