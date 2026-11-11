"""Reference FASTAs of the closed-loop runs, rebuilt from the public genomes.

Icarust samples reads from two FASTA files: ``host.fa`` (200 human 100-kb blocks) and ``target.fa`` (13 microbial
100-kb blocks). They are the held-out (TEST-role) genomic blocks of simulated group 1; the host blocks are a seeded
subset because Icarust converts every record to signal at start-up. ``reference_blocks.json`` lists every block in
record order with the source genomes. Each record is the uppercased interval [start, end) of its contig, named
``<reference id>|<contig>|<start>-<end>`` and wrapped at 80 bases; the output is checked against the SHA-256 of
the files used in the paper runs.
"""
from pathlib import Path
import gzip
import hashlib
import json

BLOCKS = Path(__file__).with_name('reference_blocks.json')


def load_blocks(path=BLOCKS):
    return json.loads(Path(path).read_text())


def _open(path):
    path = Path(path)
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def extract(fastas, wanted):
    """wanted: {contig: [(start, end), ...]} -> {(contig, start, end): uppercase sequence}.

    Streams each FASTA once and keeps only the requested intervals; contig names are the first word of the header."""
    found, pieces = set(), {(c, s, e): [] for c, spans in wanted.items() for s, e in spans}
    for fasta in fastas:
        spans, pos = None, 0
        with _open(fasta) as f:
            for line in f:
                if line.startswith('>'):
                    contig = line[1:].split()[0]
                    spans = sorted(wanted.get(contig, ()))
                    if spans:
                        if contig in found:
                            raise ValueError(f'contig {contig} occurs twice in the input FASTAs')
                        found.add(contig)
                    current, pos = contig, 0
                    continue
                if not spans:
                    continue
                seq = line.strip()
                end = pos + len(seq)
                for s, e in spans:
                    if s < end and e > pos:
                        pieces[(current, s, e)].append(seq[max(s, pos) - pos:min(e, end) - pos])
                pos = end
    missing = sorted(set(wanted) - found)
    if missing:
        raise ValueError(f'contigs not found in the input FASTAs: {", ".join(missing)}')
    out = {}
    for key, parts in pieces.items():
        seq = ''.join(parts).upper()
        if len(seq) != key[2] - key[1]:
            raise ValueError(f'{key[0]}:{key[1]}-{key[2]} is outside the contig')
        out[key] = seq
    return out


def fasta_text(records, width=80):
    """records: [(name, sequence)] -> FASTA text."""
    lines = []
    for name, seq in records:
        lines.append(f'>{name}\n')
        lines += [seq[i:i + width] + '\n' for i in range(0, len(seq), width)]
    return ''.join(lines)


def record_name(reference_id, contig, start, end):
    return f'{reference_id}|{contig}|{start}-{end}'.replace(' ', '_')


def build(fastas, out, blocks=None, check=True):
    """Write host.fa, target.fa and manifest.json to `out`. Returns the manifest.

    check: require the SHA-256 of the paper runs' files (False only for a custom block list)."""
    spec = load_blocks() if blocks is None else blocks
    out = Path(out)
    for name in ('host.fa', 'target.fa', 'manifest.json'):
        if (out/name).exists():
            raise FileExistsError(out/name)
    wanted = {}
    for rows in spec['samples'].values():
        for _, contig, start, end in rows:
            wanted.setdefault(contig, []).append((int(start), int(end)))
    seqs = extract(fastas, wanted)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(blocks=str(BLOCKS.name) if blocks is None else 'custom', inputs=[str(Path(f).resolve()) for f in fastas],
                    samples={})
    for sample, rows in spec['samples'].items():
        records = [(record_name(ref, contig, s, e), seqs[(contig, int(s), int(e))]) for ref, contig, s, e in rows]
        text = fasta_text(records, spec.get('line_width', 80))
        digest = hashlib.sha256(text.encode()).hexdigest()
        expected = spec.get('sha256', {}).get(sample)
        if check and expected and digest != expected:
            raise ValueError(f'{sample}.fa differs from the paper reference (sha256 {digest}, expected {expected}); '
                             'check the input genome versions')
        (out/f'{sample}.fa').write_text(text)
        manifest['samples'][sample] = dict(fasta=f'{sample}.fa', sha256=digest, records=len(records),
                                           bases=sum(len(s) for _, s in records), matches_paper=digest == expected)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
