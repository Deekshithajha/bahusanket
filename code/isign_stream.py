"""Stream single .pose files out of the iSign v1.1 pose archive on Hugging Face.

The archive is one Zip64 file cut into 4 byte-parts (part_aa..part_ad, 170 GB total).
We never download it: we read the zip's table of contents (about 15 MB, at the very end)
once, then fetch each .pose file with an HTTP range request (a few hundred KB each).
"""
import os, struct, zlib, pickle, requests

BASE = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/"
PARTS = ["iSign-poses_v1.1_part_aa", "iSign-poses_v1.1_part_ab",
         "iSign-poses_v1.1_part_ac", "iSign-poses_v1.1_part_ad"]
PART_SIZES = [48318382080, 48318382080, 48318382080, 25275140832]
TOTAL = sum(PART_SIZES)


def _headers(extra=None):
    h = dict(extra or {})
    tok = os.environ.get("HF_TOKEN")  # in Colab: set from userdata, never hard-code
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def read_range(start, length, session=None):
    """Read `length` bytes at global offset `start`, crossing part boundaries if needed."""
    s = session or requests
    out, pos, end = b"", start, start + length
    off = 0
    for name, size in zip(PARTS, PART_SIZES):
        lo, hi = off, off + size
        if pos < hi and end > lo:
            a, b = max(pos, lo) - lo, min(end, hi) - lo - 1
            r = s.get(BASE + name, headers=_headers({"Range": f"bytes={a}-{b}"}), timeout=120)
            r.raise_for_status()
            assert len(r.content) == b - a + 1, "short read"
            out += r.content
        off = hi
    return out


def load_index(cache="isign_zip_index.pkl", session=None):
    """Return {uid: (local_header_offset, compressed_size, method)} for every .pose file."""
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    tail = read_range(TOTAL - 200, 200, session)
    i = tail.rfind(b"PK\x06\x06")  # Zip64 end-of-central-directory record
    assert i >= 0, "no Zip64 EOCD found"
    n_entries, cd_size, cd_off = struct.unpack("<QQQ", tail[i + 32:i + 56])
    cd = read_range(cd_off, cd_size, session)
    idx, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0  # Zip64 extra field holds the real values when the 32-bit ones are 0xFFFFFFFF
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize, csize, lho):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(flag)
                usize, csize, lho = vals
            q += 4 + hl
        if name.endswith(".pose"):
            idx[name.split("/")[-1][:-5]] = (lho, csize, method)
        p += 46 + nlen + xlen + clen
    assert len(idx) > 0.99 * (n_entries - 1), (len(idx), n_entries)
    pickle.dump(idx, open(cache, "wb"))
    return idx


def fetch_pose_bytes(uid, index, session=None):
    lho, csize, method = index[uid]
    hdr = read_range(lho, 30, session)
    assert hdr[:4] == b"PK\x03\x04"
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    data = read_range(lho + 30 + nlen + xlen, csize, session)
    if method == 8:
        data = zlib.decompress(data, -15)
    return data
