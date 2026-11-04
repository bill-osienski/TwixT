"""H-TB: the one approved metadata read of twixtbot's pinned `variables.index` (2026-11-03).

Reads a TensorFlow tensor-bundle index (a LevelDB-format table whose values are serialized
`BundleEntryProto`s) with the standard library only: no TensorFlow, no protobuf package, no
weights. It prints every variable's name, dtype, shape and byte extent in the data shard,
and refuses unless the file's sha256 equals the pin our G3 harness already carries
(`twixtbot_g3_schedule.ANCHOR["variables_index_sha256"]`).

It establishes names, shapes and byte extents only. It does not establish the graph's
operations (activations, strides, padding, how tensors are wired), so it cannot prove
conversion parity or the forward function.

    python3 -I 01_read_index.py <path to variables.index>
"""
import hashlib
import json
import struct
import sys

PIN_SHA256 = "1d4073c4da30e515a984faef2c3cc9c8b8ba402353a01933c8bb80e1f279f8cd"
MAGIC = 0xDB4775248B80FB57
DTYPES = {1: "float32", 2: "float64", 3: "int32", 9: "int64", 10: "bool", 7: "string"}


def varint(b, i):
    shift = result = 0
    while True:
        x = b[i]
        i += 1
        result |= (x & 0x7F) << shift
        if not x & 0x80:
            return result, i
        shift += 7


def handle(b, i):
    off, i = varint(b, i)
    size, i = varint(b, i)
    return (off, size), i


def block(data, h):
    off, size = h
    raw = data[off:off + size]
    ctype = data[off + size]
    if ctype != 0:
        raise SystemExit(f"block at {off} has compression type {ctype}; only uncompressed is supported")
    n_restarts = struct.unpack_from("<I", raw, len(raw) - 4)[0]
    end = len(raw) - 4 - 4 * n_restarts
    out, i, last = [], 0, b""
    while i < end:
        shared, i = varint(raw, i)
        non_shared, i = varint(raw, i)
        vlen, i = varint(raw, i)
        key = last[:shared] + raw[i:i + non_shared]
        i += non_shared
        out.append((key, raw[i:i + vlen]))
        i += vlen
        last = key
    return out


def fields(msg):
    """Minimal protobuf wire decoder: {field: [values]}; length-delimited values stay bytes."""
    out, i = {}, 0
    while i < len(msg):
        tag, i = varint(msg, i)
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = varint(msg, i)
        elif wt == 1:
            v = struct.unpack_from("<Q", msg, i)[0]
            i += 8
        elif wt == 2:
            n, i = varint(msg, i)
            v = msg[i:i + n]
            i += n
        elif wt == 5:
            v = struct.unpack_from("<I", msg, i)[0]
            i += 4
        else:
            raise SystemExit(f"unsupported wire type {wt}")
        out.setdefault(fno, []).append(v)
    return out


def entry(value):
    f = fields(value)
    shape = []
    for dim in fields(f[2][0]).get(2, []) if 2 in f else []:
        shape.append(fields(dim).get(1, [0])[0])
    return {
        "dtype": DTYPES.get(f.get(1, [0])[0], f.get(1, [0])[0]),
        "shape": shape,
        "shard": f.get(3, [0])[0],
        "offset": f.get(4, [0])[0],
        "size": f.get(5, [0])[0],
    }


def main(path):
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    if sha != PIN_SHA256:
        raise SystemExit(f"REFUSED: sha256 {sha} != pin {PIN_SHA256}")
    if struct.unpack_from("<Q", data, len(data) - 8)[0] != MAGIC:
        raise SystemExit("REFUSED: not a LevelDB-format table (bad magic)")
    footer = data[len(data) - 48:]
    _meta, i = handle(footer, 0)
    index_h, _ = handle(footer, i)
    rows = []
    for _k, hv in block(data, index_h):
        h, _ = handle(hv, 0)
        for key, value in block(data, h):
            if key == b"":
                continue          # the BundleHeaderProto
            rows.append({"name": key.decode(), **entry(value)})
    total = sum(r["size"] for r in rows)
    print(json.dumps({"sha256": sha, "n_variables": len(rows), "data_bytes_sum": total,
                      "variables": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
