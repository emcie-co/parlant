import hashlib


def md5_checksum(input: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode("utf-8"))

    return md5_hash.hexdigest()
