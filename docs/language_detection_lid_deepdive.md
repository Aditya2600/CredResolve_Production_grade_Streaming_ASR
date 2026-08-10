# Language Detection (LID) Deep-Dive

This document provides a comprehensive overview of the Language Detection (LID) implementation in the `worker` service of the CredResolve ASR platform.

## Overview

Language Detection (LID) is implemented within the audio processing pipeline to automatically identify the spoken language of incoming audio streams. It is crucial for downstream processes like context biasing, correct ASR decoding, and Inverse Text Normalization (ITN).

The core of the LID system is designed to be **modular**, **extensible**, and **fault-tolerant**, supporting multiple machine learning providers and a configurable fallback mechanism.

The primary implementation is located in [`worker/app/lid.py`](file:///Users/aditya/CredResolve_Production_grade_Streaming_ASR/worker/app/lid.py).

## Providers and Implementations

The system abstracts all language detectors behind the `BaseLanguageDetector` interface. There are currently two main implementations:

### 1. SpeechBrain / VoxLingua107
- **Class**: `SpeechBrainLanguageDetector`
- **Supported Providers**: `speechbrain`, `voxlingua`, `voxlingua107`
- **Backend**: Uses `speechbrain.pretrained.EncoderClassifier` (or `speechbrain.inference.classifiers.EncoderClassifier`).
- **Characteristics**: This is typically based on the robust VoxLingua107 dataset, capable of detecting over 100 languages. 
- **Inference**: Configured to run on the CPU to prevent GPU memory fragmentation alongside heavy ASR models.

### 2. Vakgyata
- **Class**: `VakgyataLanguageDetector`
- **Supported Providers**: `vakgyata`
- **Backend**: Uses Hugging Face Transformers (`AutoFeatureExtractor` and `AutoModelForAudioClassification`).
- **Characteristics**: Geared toward specialized or regional Indian languages.
- **Inference**: Also runs on CPU for consistent deployment and resource allocation.

## The Fallback Chain Mechanism

To maximize accuracy and resilience, the LID system implements a **Fallback Chain** via the `FallbackLanguageDetector` class. This allows the system to cascade through multiple detectors if the primary one fails or yields low confidence.

### How it works:
1. **Primary Evaluation**: The audio is first fed into the configured `primary` detector.
2. **Confidence Check**: If the primary detector returns a result with a confidence score *below* a configured `confidence_threshold` (e.g., 0.70), a fallback is triggered.
3. **Mappable Label Check**: If the primary detector returns a raw label that cannot be mapped to any of the supported system languages (defined in `_ALIAS_MAP`), a fallback is triggered.
4. **Exception Handling**: If the primary detector crashes or throws an exception, the system automatically swallows the error and triggers the fallback.
5. **Fallback Evaluation**: If triggered, the `fallback` detector is used to process the audio.
6. **Result Merging**: The final `DetectionResult` encapsulates both the fallback result (as the final language) and the original primary attempt (metadata, original confidence, and reason for fallback), aiding in observability and debugging.

## Data Normalization

Raw predictions from different models (e.g., `en-in`, `english`, `hin`) are normalized into standard ISO language codes using an internal `_ALIAS_MAP`. 

Examples:
- `bengali`, `ben`, `bn-in` $\rightarrow$ `bn`
- `english`, `eng`, `en-in`, `en-us` $\rightarrow$ `en`
- `hindi`, `hin`, `hi-in` $\rightarrow$ `hi`

This ensures that downstream systems receive a predictable, unified language code regardless of the underlying ML provider.

## Input Constraints

The LID module enforces strict audio input requirements:
- **Sample Rate**: Must be `16000 Hz` (16 kHz).
- **Format**: `PCM 16-bit` audio bytes.
- The pipeline automatically converts these bytes into 32-bit floats (`float32`) normalized between `[-1.0, 1.0]` before passing them to the acoustic models.

## Integration in `worker/app/model.py`

In the worker runtime environment, LID is initialized and enabled based on environmental variables or server configuration:
- `lid_primary_provider` / `lid_primary_source`
- `lid_fallback_provider` / `lid_fallback_source`
- `lid_confidence_threshold`

During the model initialization phase (`_initialize_lid`), the `build_language_detector` factory function links the primary and fallback configurations, loads the models onto the CPU, and prepares them for real-time inference on incoming audio frames.
