"""Converts the mecab-ipadic 2.7.0 source dictionary into the compact files read by SubVox Pro.

Usage: python convert_ipadic.py <mecab-ipadic source dir> <output dir>

Outputs (all UTF-8):
  lexicon.tsv.gz  surface, left id, right id, cost, reading (katakana, empty when unknown)
  matrix.bin.gz   int32 left size, int32 right size, then int16 costs indexed [left + left_size * right]
  char.def        character categories, re-encoded from EUC-JP
  unk.def         unknown-word entries, re-encoded from EUC-JP
  COPYING         the dictionary license, re-encoded from EUC-JP
"""
import glob
import gzip
import os
import struct
import sys

source, output = sys.argv[1], sys.argv[2]
os.makedirs(output, exist_ok=True)


def read_euc(name, errors="strict"):
    with open(os.path.join(source, name), "rb") as handle:
        return handle.read().decode("euc_jp", errors)


rows = 0
with gzip.open(os.path.join(output, "lexicon.tsv.gz"), "wt", encoding="utf-8", newline="\n", compresslevel=9) as lexicon:
    for path in sorted(glob.glob(os.path.join(source, "*.csv"))):
        for line in read_euc(os.path.basename(path)).splitlines():
            fields = line.split(",")
            if len(fields) < 4:
                continue
            reading = fields[11] if len(fields) > 11 and fields[11] != "*" else ""
            lexicon.write("\t".join([fields[0], fields[1], fields[2], fields[3], reading]) + "\n")
            rows += 1

matrix_lines = read_euc("matrix.def").splitlines()
left_size, right_size = map(int, matrix_lines[0].split())
costs = [0] * (left_size * right_size)
for line in matrix_lines[1:]:
    if line.strip():
        left, right, cost = map(int, line.split())
        costs[left + left_size * right] = cost
assert all(-32768 <= cost <= 32767 for cost in costs)
with gzip.open(os.path.join(output, "matrix.bin.gz"), "wb", compresslevel=9) as matrix:
    matrix.write(struct.pack("<ii", left_size, right_size))
    matrix.write(struct.pack("<%dh" % len(costs), *costs))

for name in ("char.def", "unk.def", "COPYING"):
    with open(os.path.join(output, name), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(read_euc(name, "replace").replace("�", "").rstrip() + "\n")

print("entries", rows, "matrix", left_size, right_size)
