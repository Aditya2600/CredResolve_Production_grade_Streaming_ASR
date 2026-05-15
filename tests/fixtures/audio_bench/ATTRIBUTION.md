# Audio bench fixtures — attribution

These four WAV files are excerpts from the **Vaani** speech corpus by
ARTPARK @ IISc Bangalore, redistributed under the original
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Source

- Dataset: `ARTPARK-IISc/Vaani-transcription-part` (HuggingFace Hub, gated)
- Config: `audio/Hindi`
- Split: `train`
- Snapshot fetched: 2026-05-15 (UTC)
- Selection: deterministic, seed=20250507, scan_limit=400, n=4, duration ∈ [2.0, 5.0] s
- Audio decoded to 16 kHz mono PCM16 WAV (no other processing).

## Citation

ARTPARK-IISc, *Vaani — A bilingual, code-mixed Indian speech dataset*. https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part

## Selected clips

| WAV | Vaani index | Duration (s) | Transcript |
|---|---|---|---|
| `01_iisc_vaaniproject_m_biha.wav` | 73 | 2.24 | विद्यापति आश्रम है |
| `02_iisc_vaaniproject_m_biha.wav` | 120 | 2.12 | घर नजर आ रहा है |
| `03_iisc_vaaniproject_k_utta.wav` | 259 | 3.966 | यह लाल रंग की एक किताब है जो कि पंडित लोग |
| `04_iisc_vaaniproject_k_jhar.wav` | 292 | 3.158 | कुछ लोग क्रिकेट खेल रहे है |

Refer to `_fetch_vaani_fixtures.py` to regenerate this set.
