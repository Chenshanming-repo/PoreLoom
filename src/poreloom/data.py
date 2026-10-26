"""File-integrity helper for signal caches and checkpoint provenance."""
import hashlib


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
