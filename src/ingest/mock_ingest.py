# ingest/mock_ingest.py
import pandas as pd
import json
from datetime import datetime
import logging
import os
import sys

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename='logs/ingest.log', 
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

INPUT_CSV = 'data/sample_reports.csv'
OUTPUT_GEOJSON = 'data/raw_reports.geojson'

def validate_row(row):
    """
    Validates a single row of citizen report data.
    Returns (True, None) if valid, or (False, error_message) if invalid.
    """
    try:
        # Check if description exists and is not empty
        if pd.isna(row['description']) or str(row['description']).strip() == '':
            return False, 'empty description'
        
        # Validate latitude
        lat = float(row['lat'])
        if not (-90 <= lat <= 90):
            return False, 'latitude out of range (-90 to 90)'
        
        # Validate longitude
        lon = float(row['lon'])
        if not (-180 <= lon <= 180):
            return False, 'longitude out of range (-180 to 180)'
        
        # Validate timestamp format
        timestamp_str = str(row['timestamp']).replace('Z', '+00:00')
        datetime.fromisoformat(timestamp_str)
        
        return True, None
        
    except ValueError as e:
        return False, f'value error: {str(e)}'
    except Exception as e:
        return False, f'validation error: {str(e)}'

def create_geojson_feature(row):
    """
    Creates a GeoJSON feature from a validated row.
    """
    feature = {
        'type': 'Feature',
        'properties': {
            'id': int(row['id']),
            'description': str(row['description']),
            'timestamp': str(row['timestamp']),
            'source': str(row.get('source', 'citizen')),
            'image_path': str(row['image_path']) if pd.notna(row['image_path']) else None
        },
        'geometry': {
            'type': 'Point',
            'coordinates': [float(row['lon']), float(row['lat'])]
        }
    }
    return feature

if __name__ == '__main__':
    print("Starting ingestion process...")
    print(f"Reading from: {INPUT_CSV}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file {INPUT_CSV} not found!")
        logging.error(f"Input file {INPUT_CSV} not found")
        exit(1)
    
    # Read CSV file
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} rows from CSV")
        logging.info(f"Loaded {len(df)} rows from {INPUT_CSV}")
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        logging.error(f"Failed to read CSV: {e}")
        exit(1)
    
    # Process each row
    features = []
    valid_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        row_id = row.get('id', f'unknown_{idx}')
        
        # Validate the row
        is_valid, error_msg = validate_row(row)
        
        if not is_valid:
            error_count += 1
            logging.error(f"id={row_id} status=ERROR msg={error_msg}")
            print(f"  [X] Row {row_id}: {error_msg}")
            continue
        
        # Create GeoJSON feature
        try:
            feature = create_geojson_feature(row)
            features.append(feature)
            valid_count += 1
            logging.info(f"id={row_id} status=OK")
            print(f"  [OK] Row {row_id}: Valid")
        except Exception as e:
            error_count += 1
            logging.error(f"id={row_id} status=ERROR msg=feature creation failed: {e}")
            print(f"  [X] Row {row_id}: feature creation failed")
    
    # Create GeoJSON FeatureCollection
    geojson_output = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Write to output file
    try:
        with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
            json.dump(geojson_output, f, ensure_ascii=False, indent=2)
        print(f"\n[SUCCESS] Wrote {valid_count} features to {OUTPUT_GEOJSON}")
        logging.info(f"Successfully wrote {valid_count} features to {OUTPUT_GEOJSON}")
    except Exception as e:
        print(f"\n[ERROR] writing GeoJSON: {e}")
        logging.error(f"Failed to write GeoJSON: {e}")
        exit(1)
    
    # Summary
    print(f"\n=== Ingestion Summary ===")
    print(f"Total rows processed: {len(df)}")
    print(f"Valid features: {valid_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {OUTPUT_GEOJSON}")
    print(f"Log file: logs/ingest.log")
    
    logging.info(f"Ingestion complete: {valid_count} valid, {error_count} errors")