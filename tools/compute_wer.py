#!/usr/bin/env python3
import argparse
from pathlib import Path


def edit_distance(a: list[str], b: list[str]) -> tuple[int, int, int]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    op = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        op[i][0] = 'D'
    for j in range(1, n + 1):
        dp[0][j] = j
        op[0][j] = 'I'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = 'M'
                continue

            sub = dp[i - 1][j - 1] + 1
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            best = min(sub, delete, insert)
            dp[i][j] = best
            if best == sub:
                op[i][j] = 'S'
            elif best == delete:
                op[i][j] = 'D'
            else:
                op[i][j] = 'I'

    i, j = m, n
    s = d = ins = 0
    while i > 0 or j > 0:
        current = op[i][j]
        if current in {'M', 'S'}:
            if current == 'S':
                s += 1
            i -= 1
            j -= 1
        elif current == 'D':
            d += 1
            i -= 1
        elif current == 'I':
            ins += 1
            j -= 1
        else:
            break

    return s, d, ins


def normalize(text: str) -> list[str]:
    return text.strip().split()


def load_pairs_from_tsv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not raw.strip():
            continue
        if '\t' not in raw:
            raise SystemExit(f'{path}:{lineno}: expected TAB-separated <reference> <hypothesis>')
        ref, hyp = raw.split('\t', 1)
        pairs.append((ref, hyp))
    return pairs


def load_pairs_from_files(ref_path: Path, hyp_path: Path) -> list[tuple[str, str]]:
    refs = [line.rstrip('\n') for line in ref_path.read_text(encoding='utf-8').splitlines()]
    hyps = [line.rstrip('\n') for line in hyp_path.read_text(encoding='utf-8').splitlines()]
    if len(refs) != len(hyps):
        raise SystemExit(
            f'mismatched line counts: {ref_path} has {len(refs)} lines, '
            f'{hyp_path} has {len(hyps)} lines'
        )
    return list(zip(refs, hyps))


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute corpus WER from reference/hypothesis text pairs.')
    parser.add_argument('--tsv', type=Path, help='TSV file with: reference<TAB>hypothesis per line')
    parser.add_argument('--refs', type=Path, help='Reference text file, one utterance per line')
    parser.add_argument('--hyps', type=Path, help='Hypothesis text file, one utterance per line')
    args = parser.parse_args()

    if args.tsv:
        if args.refs or args.hyps:
            raise SystemExit('use either --tsv or --refs/--hyps')
        pairs = load_pairs_from_tsv(args.tsv)
    else:
        if not args.refs or not args.hyps:
            raise SystemExit('provide --tsv or both --refs and --hyps')
        pairs = load_pairs_from_files(args.refs, args.hyps)

    total_s = total_d = total_i = total_words = 0
    for ref, hyp in pairs:
        ref_words = normalize(ref)
        hyp_words = normalize(hyp)
        s, d, i = edit_distance(ref_words, hyp_words)
        total_s += s
        total_d += d
        total_i += i
        total_words += len(ref_words)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0
    print(f'utterances={len(pairs)}')
    print(f'reference_words={total_words}')
    print(f'substitutions={total_s}')
    print(f'deletions={total_d}')
    print(f'insertions={total_i}')
    print(f'wer={wer:.4f}')
    print(f'wer_percent={wer * 100:.2f}')


if __name__ == '__main__':
    main()
