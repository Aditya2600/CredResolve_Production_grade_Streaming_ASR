import csv, os, wave, contextlib, random, sys
base='/Users/aditya/CredResolve_Production_grade_Streaming_ASR'
F1=os.path.join(base,'data_correction/Dataset generation - batch_transcript-parent 2 - Sheet 1 - batch_transcript-pare-2.csv')
F2=os.path.join(base,'data_correction/Transcription Phase 2 - Transcription Phase 2 - batch_transcript (3) (2).csv')
DIR1=os.path.join(base,'vad_chunks')
DIR2=os.path.join(base,'vad_chunks/transcription level2')

def dur(p):
    with contextlib.closing(wave.open(p)) as w:
        return w.getnframes()/w.getframerate()

pool=[]  # (fname, ref=ground_truth, hyp=indic_conformer_transcript, wavpath, src)

# F1: exclude 3 ranges, keep rest (incl drop remarks). cols: filename,audio_link,language_id,transcript,ground_truth_transcript,Remarks
data=list(csv.reader(open(F1)))[2:]
excl=set(range(1120,1149))|set(range(1202,1319))|set(range(1353,1399))
for i,r in enumerate(data,1):
    if i in excl or len(r)<5 or not r[0]: continue
    fn,hyp,ref=r[0],r[3],r[4]
    wp=os.path.join(DIR1,fn)
    if not os.path.exists(wp): continue
    if not (ref and ref.strip()): continue
    pool.append((fn,ref,hyp,wp,'F1'))

# F2: remark in {Done,Drop,Perfect}. cols: filename,...,transcript(idx3),Ground Truth(idx4),...,Remarks(idx7)
for r in csv.DictReader(open(F2)):
    rem=(r.get('Remarks') or '').strip()
    if rem not in ('Done','Drop','Perfect'): continue
    fn=r['filename']; hyp=r['transcript']; ref=r['Ground Truth']
    wp=os.path.join(DIR2,fn)
    if not os.path.exists(wp): continue
    if not (ref and ref.strip()): continue
    pool.append((fn,ref,hyp,wp,'F2'))

nf1=sum(1 for x in pool if x[4]=='F1'); nf2=sum(1 for x in pool if x[4]=='F2')
print(f'pool eligible: {len(pool)}  (F1={nf1}, F2={nf2})',file=sys.stderr)

# shuffle, accumulate to 1.5h = 5400s
random.seed(42)
random.shuffle(pool)
BUDGET=1.5*3600
sel=[]; acc=0.0
for fn,ref,hyp,wp,src in pool:
    d=dur(wp)
    if acc+d>BUDGET and sel: break
    sel.append((fn,ref,hyp,src,d)); acc+=d

sf1=sum(1 for x in sel if x[3]=='F1'); sf2=sum(1 for x in sel if x[3]=='F2')
print(f'test set: {len(sel)} utts, {acc/3600:.3f}h  (F1={sf1}, F2={sf2})',file=sys.stderr)

out=os.path.join(base,'data_correction/testset_1p5h.tsv')
with open(out,'w') as f:
    for fn,ref,hyp,src,d in sel:
        r=' '.join((ref or '').split()); h=' '.join((hyp or '').split())
        f.write(f'{r}\t{h}\n')
print(out)
# also manifest for traceability
man=os.path.join(base,'data_correction/testset_1p5h_manifest.csv')
with open(man,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['filename','source','duration_s','ground_truth','indic_conformer'])
    for fn,ref,hyp,src,d in sel: w.writerow([fn,src,f'{d:.2f}',ref,hyp])
print(man)
