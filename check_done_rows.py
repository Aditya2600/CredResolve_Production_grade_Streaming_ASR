import json, csv
from pathlib import Path

jobs_dir = Path('outputs/smfg_rural_sheet4_sarvam/sarvam_jobs')
results_dir = Path('outputs/smfg_rural_sheet4_sarvam/sarvam_results')

transcribed_urls = set()
for batch_file in sorted(jobs_dir.glob('batch_*.json')):
    state = json.loads(batch_file.read_text())
    job_id = state.get('job_id', '')
    job_status = (state.get('status') or {}).get('job_state', 'unknown')
    if job_status != 'Completed':
        continue
    for item in state.get('files', []):
        result_name = item.get('result_name', '')
        result_path = results_dir / job_id / result_name
        if result_path.is_file() and result_path.stat().st_size > 0:
            transcribed_urls.add(item['url'])

print(f'Unique URLs with transcripts: {len(transcribed_urls)}')

source_csv = Path('recordings/smfg_rural_with_recording - Sheet4.csv')
with source_csv.open(encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

url_col = 'cr_recording_url'
done_rows = []
for i, row in enumerate(rows, start=2):
    url = row.get(url_col, '').strip()
    if url in transcribed_urls:
        done_rows.append(i)

print(f'Total source CSV rows: {len(rows)}')
print(f'Rows WITH transcripts: {len(done_rows)}')
if done_rows:
    print(f'CSV line range: {min(done_rows)} to {max(done_rows)}')
    print('Done row line numbers:', done_rows)
