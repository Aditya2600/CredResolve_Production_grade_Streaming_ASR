import pandas as pd
import soundfile as sf
import os
import json
from pathlib import Path

def analyze():
    df = pd.read_csv('akshay-docs/inventory/data_inventory.csv')
    durations = []
    for p in df['wav_audio_path']:
        if isinstance(p, str) and os.path.exists(p):
            durations.append(sf.info(p).duration)
        else:
            durations.append(0.0)
            
    df['audio_duration'] = durations
    print("\n--- Duration Statistics ---")
    print(df['audio_duration'].describe())
    
    n_in_range = ((df['audio_duration'] >= 3.0) & (df['audio_duration'] <= 35.0)).sum()
    print(f"\nSamples in 3.0s - 35.0s range: {n_in_range} out of {len(df)}")
    
    # Check JSON join logic
    sample_json = df.loc[0, 'native_language_transcript']
    data = json.loads(sample_json)
    turns = [t.get('text') or t.get('en_text', '') for t in data.get('interaction_transcript', [])]
    merged = " ".join(turns).strip()
    print(f"\nSample Merged Transcript:\n{merged}")

if __name__ == "__main__":
    analyze()
