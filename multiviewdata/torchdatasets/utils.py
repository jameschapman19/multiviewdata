import gzip


def fopen(p, flag="r"):
    """Opens as gzipped if appropriate, else as ascii."""
    if p.endswith(".gz"):
        if "w" in flag:
            return gzip.open(p, "wb")
        else:
            return gzip.open(p, "rt")
    else:
        return open(p, flag)
