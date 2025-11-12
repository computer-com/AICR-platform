"""
Generate More Sample Crisis Reports Data
Creates a larger dataset with 100+ crisis reports for better model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed
np.random.seed(42)
random.seed(42)

print("="*60)
print("Generating Enhanced Crisis Reports Dataset")
print("="*60)

# Crisis type templates
crisis_templates = {
    'flood': [
        'Water rising rapidly on {street}. Roads completely flooded.',
        'Severe flooding reported at {location}. Evacuation needed.',
        'Flash flood warning. Water knee-deep near {location}.',
        'Flooded basement on {street}. Water damage severe.',
        'River overflowing. Flooding affecting {location} area.',
        'Heavy rain causing flooding on {street}.',
        'Storm drains overwhelmed. Flooding at {location}.',
        'Water main break causing flooding near {location}.',
        'Residential area flooded. Multiple homes affected at {street}.',
        'Emergency flood situation at {location}. Immediate help needed.'
    ],
    'fire': [
        'Fire broke out in building on {street}. Smoke visible.',
        'Multiple alarm fire at {location}. Several floors affected.',
        'Residential fire reported on {street}. Flames spreading.',
        'Commercial building fire at {location}. Heavy smoke.',
        'Wildfire approaching residential area near {location}.',
        'Vehicle fire blocking {street}. Emergency crews responding.',
        'Electrical fire reported at {location}. Power outage.',
        'Kitchen fire at {street}. Smoke alarm activated.',
        'Fire spreading to adjacent properties near {location}.',
        'Smoke visible from {street}. Fire department notified.'
    ],
    'earthquake': [
        'Strong earthquake felt across {location}. Buildings shaking.',
        'Earthquake reported. Tremors felt at {street}.',
        'Seismic activity detected near {location}. Aftershocks expected.',
        'Building damage after earthquake at {street}.',
        'Ground shaking reported. People evacuating from {location}.',
        'Major earthquake. Structural damage visible at {street}.',
        'Aftershock felt near {location}. Residents concerned.',
        'Earthquake caused cracks in building at {street}.',
        'Tremors continuing at {location}. Emergency assessment needed.',
        'Seismic event. Windows shattered at {street}.'
    ],
    'gas_leak': [
        'Gas leak detected near {location}. Area evacuated.',
        'Strong gas odor reported on {street}. Emergency response.',
        'Natural gas leak at {location}. Residents evacuating.',
        'Gas line rupture on {street}. Hazmat team responding.',
        'Gas smell reported near {location}. Fire department on scene.',
        'Pipeline leak detected at {street}. Area cordoned off.',
        'Gas explosion risk at {location}. Immediate evacuation.',
        'Utility crews responding to gas leak on {street}.',
        'Suspected gas leak at {location}. Businesses closed.',
        'Emergency gas shutoff needed at {street}.'
    ],
    'landslide': [
        'Landslide blocking {street}. Road closed.',
        'Mudslide reported near {location}. Heavy debris.',
        'Hillside collapse at {street}. Homes threatened.',
        'Landslide after heavy rain near {location}.',
        'Rock slide blocking highway at {street}.',
        'Debris flow affecting {location} area.',
        'Slope failure on {street}. Road impassable.',
        'Mudslide damage to property at {location}.',
        'Landslide risk increasing near {street}.',
        'Hillside erosion causing slide at {location}.'
    ],
    'power_outage': [
        'Power outage affecting entire {location} neighborhood.',
        'Electrical grid failure on {street}. No power.',
        'Transformer explosion. Power out at {location}.',
        'Wide area blackout near {street}. Cause unknown.',
        'Storm knocked out power at {location}.',
        'Power lines down on {street}. Outage reported.',
        'Substation failure causing outage at {location}.',
        'Rolling blackouts affecting {street} area.',
        'Power restored partially at {location}.',
        'Extended outage expected for {street} residents.'
    ],
    'storm': [
        'Severe storm causing damage on {street}.',
        'High winds and heavy rain at {location}.',
        'Storm knocked down trees near {street}.',
        'Tornado warning for {location} area.',
        'Severe thunderstorm with hail at {street}.',
        'Storm damage widespread in {location}.',
        'Hurricane force winds reported near {street}.',
        'Lightning strike caused fire at {location}.',
        'Storm surge flooding coastal areas near {street}.',
        'Blizzard conditions making {location} roads impassable.'
    ]
}

# Location templates
streets = [
    'Main Street', 'Elm Street', 'Oak Avenue', 'Maple Road', 'Pine Street',
    'Cedar Lane', 'First Avenue', 'Second Street', 'Broadway', 'Park Avenue',
    'Church Street', 'School Road', 'Mill Street', 'River Road', 'Lake Drive',
    'Hill Street', 'Valley Road', 'Mountain View', 'Forest Avenue', 'Garden Street',
    'Washington Street', 'Lincoln Avenue', 'Jefferson Road', 'Madison Lane', 'Monroe Street'
]

locations = [
    'downtown', 'city center', 'north district', 'south end', 'east side',
    'west quarter', 'industrial park', 'residential area', 'commercial district', 'suburb',
    'waterfront', 'historic district', 'business park', 'shopping center', 'community center'
]

# Toronto area coordinates (varied)
base_lat = 43.6532
base_lon = -79.3832

def generate_coordinates():
    """Generate random coordinates around Toronto"""
    lat = base_lat + np.random.uniform(-0.05, 0.05)
    lon = base_lon + np.random.uniform(-0.05, 0.05)
    return round(lat, 4), round(lon, 4)

def generate_timestamp(start_days_ago=30):
    """Generate random timestamp within last N days"""
    days_ago = random.randint(0, start_days_ago)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    dt = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Generate dataset
print("\nGenerating crisis reports...")
data = []
record_id = 1

# Generate 15-20 reports per crisis type
for crisis_type, templates in crisis_templates.items():
    num_reports = random.randint(15, 20)
    
    for i in range(num_reports):
        # Select random template and location
        template = random.choice(templates)
        location = random.choice(locations)
        street = random.choice(streets)
        
        # Fill template
        description = template.format(location=location, street=street)
        
        # Generate coordinates and timestamp
        lat, lon = generate_coordinates()
        timestamp = generate_timestamp()
        
        # Random source
        source = random.choice(['citizen', 'citizen', 'social'])  # More citizen reports
        
        # Random image path (some have images, some don't)
        image_path = f'images/{crisis_type}_{i}.jpg' if random.random() > 0.4 else ''
        
        data.append({
            'id': record_id,
            'description': description,
            'crisis_type': crisis_type,
            'latitude': lat,
            'longitude': lon,
            'timestamp': timestamp,
            'source': source,
            'image_path': image_path
        })
        
        record_id += 1

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Update IDs after shuffle
df['id'] = range(1, len(df) + 1)

print(f"[OK] Generated {len(df)} crisis reports")
print(f"\nCrisis type distribution:")
print(df['crisis_type'].value_counts())

# Save to CSV
output_path = 'data/mock_crisis_reports.csv'
df.to_csv(output_path, index=False)
print(f"\n[OK] Saved to {output_path}")

print("\n" + "="*60)
print("Dataset Generation Complete")
print("="*60)
print(f"\nTotal records: {len(df)}")
print(f"Crisis types: {df['crisis_type'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nNow run: python src/preprocess/eda_nlp_prep.py")
print("="*60)