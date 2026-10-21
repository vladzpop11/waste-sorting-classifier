"""Unusual and hostile uploads for checking the browser demo against Python (web_fixtures).

Each case is a file the page must treat exactly like the API: the same pixels, or refused.
They cover what real uploads contain (EXIF orientation in every place Pillow reads it,
transparency, 16-bit and low-bit-depth PNG, progressive and MPO JPEG) and what hostile or
damaged files contain (impossible chunk sizes, truncation, bad checksums).
"""

import io
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

# Files the page knowingly handles differently from Python: name -> (what the page does,
# why). The checks assert exactly this behaviour, so any change is noticed. Valid files:
# CMYK JPEG and animated WebP. All the others break their format's specification; the page
# stays safe with them but does not copy every Pillow quirk.
KNOWN_DIFFERENCES = {
    "cmyk.jpg": ("refuses", "the WebAssembly JPEG decoder does not convert CMYK"),
    "animated.webp": ("refuses", "the WebAssembly WebP decoder does not read animations"),
    "fuzz-exif-cannot-be-rewritten.jpg": (
        "decodes",
        "after rotating, Pillow writes the EXIF back and fails on values that do not fit "
        "their tag's type (Pillow refuses it)",
    ),
    "fuzz-jpeg-dpi-from-exif-indexerror.jpg": (
        "decodes",
        "a one-character EXIF resolution, which Pillow fails to read as a dpi (it refuses it)",
    ),
    "fuzz-jpeg-icc-empty-profile.jpg": (
        "refuses",
        "an ICC segment with no profile data stops the WebAssembly decoder (Pillow accepts it)",
    ),
    "fuzz-jpeg-no-eoi-padding.jpg": ("refuses", "no end-of-image marker (Pillow accepts it)"),
    "fuzz-png-bad-adler.png": (
        "decodes",
        "a wrong zlib checksum after complete image data (Pillow refuses it when it reads "
        "the checksum together with the last row)",
    ),
    "fuzz-png-compression-byte-1.png": ("refuses", "unknown compression method in IHDR"),
    "fuzz-png-no-plte.png": ("refuses", "palette image without a palette"),
    "fuzz-png-trns-after-idat.png": ("differs", "tRNS after the image data (Pillow applies it)"),
    "fuzz-png-zlib-no-adler.png": ("refuses", "image data stream without its checksum"),
    "fuzz-webp-corrupt-lossy.webp": (
        "differs",
        "corrupt lossy data decodes to a few different pixels (libwebp version)",
    ),
}


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def _tiff(entries: list[tuple[int, int, int, bytes]], big_endian: bool = True,
          extra: bytes = b"") -> bytes:  # fmt: skip
    """A TIFF block with one IFD: (tag, type, count, 4-byte value field) entries."""
    e = ">" if big_endian else "<"
    head = (b"MM\x00\x2a" if big_endian else b"II\x2a\x00") + struct.pack(e + "I", 8)
    ifd = struct.pack(e + "H", len(entries))
    for tag, typ, count, value in entries:
        ifd += struct.pack(e + "HHI", tag, typ, count) + value
    return head + ifd + b"\x00\x00\x00\x00" + extra


def _short(v: int, big_endian: bool = True) -> bytes:
    return struct.pack(">H" if big_endian else "<H", v) + b"\x00\x00"


def _exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def _xmp(orientation: int) -> bytes:
    return (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
            b'tiff:Orientation="%d"/></rdf:RDF></x:xmpmeta>' % orientation)  # fmt: skip


def _jpeg_with_segments(jpeg: bytes, segments: list[tuple[int, bytes]]) -> bytes:
    """Insert APPn segments right after SOI."""
    inserted = b"".join(bytes([0xFF, m]) + struct.pack(">H", len(p) + 2) + p for m, p in segments)
    return jpeg[:2] + inserted + jpeg[2:]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png_insert(png: bytes, chunks: bytes) -> bytes:
    """Insert chunks right after IHDR (8-byte signature + 25-byte IHDR chunk)."""
    return png[:33] + chunks + png[33:]


ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
         (0, 1, 1, 2)]  # fmt: skip


def raw_png(samples: np.ndarray, color_type: int, depth: int, interlace: bool = False,
            plte: bytes | None = None, trns: bytes | None = None) -> bytes:  # fmt: skip
    """A PNG of any colour type and bit depth from integer samples (H, W, channels), with
    filters cycling through all five types. Pillow cannot write most of these."""
    h, w = samples.shape[:2]

    def pack_rows(a: np.ndarray) -> list[bytes]:
        rows, prev = [], None
        for y, row in enumerate(a):
            flat = row.reshape(-1).astype(np.uint32)
            if depth == 16:
                data = b"".join(struct.pack(">H", int(v)) for v in flat)
            elif depth == 8:
                data = bytes(int(v) for v in flat)
            else:
                bits = "".join(format(int(v), f"0{depth}b") for v in flat)
                bits += "0" * (-len(bits) % 8)
                data = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))
            prev = prev if prev is not None else bytes(len(data))
            rows.append(_filter(data, prev, y % 5, max(1, samples.shape[2] * depth // 8)))
            prev = data
        return rows

    if interlace:
        rows = []
        for x0, y0, dx, dy in ADAM7:
            part = samples[y0::dy, x0::dx]
            if part.size:
                rows += pack_rows(part)
    else:
        rows = pack_rows(samples)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, int(interlace))
    body = _chunk(b"IHDR", ihdr)
    if plte is not None:
        body += _chunk(b"PLTE", plte)
    if trns is not None:
        body += _chunk(b"tRNS", trns)
    body += _chunk(b"IDAT", zlib.compress(b"".join(rows))) + _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + body


def _filter(data: bytes, prev: bytes, kind: int, bpp: int) -> bytes:
    """Encode one row with PNG filter `kind` (0-4) against the previous row `prev`."""
    out = bytearray([kind])
    for i, v in enumerate(data):
        a = data[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if kind == 0:
            pred = 0
        elif kind == 1:
            pred = a
        elif kind == 2:
            pred = b
        elif kind == 3:
            pred = (a + b) >> 1
        else:
            p_ = a + b - c
            pa, pb, pc = abs(p_ - a), abs(p_ - b), abs(p_ - c)
            pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
        out.append((v - pred) & 0xFF)
    return bytes(out)


def png_matrix(photo: Image.Image) -> dict[str, bytes]:
    """Every PNG colour type and bit depth, plain and interlaced, plus tRNS variants."""
    crop = photo.crop((100, 80, 137, 103))  # 37 x 23: odd sizes exercise interlacing
    rgb = np.asarray(crop, dtype=np.uint32)
    grey = np.asarray(crop.convert("L"), dtype=np.uint32)[..., None]
    noise = (np.arange(grey.size).reshape(grey.shape) * 37) & 0xFF  # low bytes of 16-bit
    alpha = np.asarray(Image.linear_gradient("L").resize(crop.size), dtype=np.uint32)[..., None]
    cases = {}
    for interlace in (False, True):
        tag = "-i" if interlace else ""
        for depth in (1, 2, 4, 8, 16):
            if depth == 16:
                cases[f"grey16{tag}.png"] = raw_png((grey << 8) | noise, 0, 16, interlace)
            else:
                cases[f"grey{depth}{tag}.png"] = raw_png(grey >> (8 - depth), 0, depth, interlace)
        cases[f"rgb8{tag}.png"] = raw_png(rgb, 2, 8, interlace)
        cases[f"rgb16{tag}.png"] = raw_png((rgb << 8) | noise, 2, 16, interlace)
        cases[f"greya8{tag}.png"] = raw_png(np.concatenate([grey, alpha], 2), 4, 8, interlace)
        cases[f"greya16{tag}.png"] = raw_png(
            np.concatenate([(grey << 8) | noise, (alpha << 8) | noise], 2), 4, 16, interlace
        )
        cases[f"rgba8{tag}.png"] = raw_png(np.concatenate([rgb, alpha], 2), 6, 8, interlace)
        cases[f"rgba16{tag}.png"] = raw_png(
            np.concatenate([(rgb << 8) | noise, (alpha << 8) | noise], 2), 6, 16, interlace
        )
        for depth in (1, 2, 4, 8):
            pal = crop.quantize(2**depth)
            idx = np.asarray(pal, dtype=np.uint32)[..., None]
            plte = bytes(pal.getpalette()[: 3 * 2**depth])
            cases[f"palette{depth}{tag}.png"] = raw_png(idx, 3, depth, interlace, plte)
    first = grey[0, 0, 0]
    cases["grey8-trns.png"] = raw_png(grey, 0, 8, trns=struct.pack(">H", first))
    cases["grey2-trns.png"] = raw_png(grey >> 6, 0, 2, trns=struct.pack(">H", 1))
    cases["grey1-trns.png"] = raw_png(grey >> 7, 0, 1, trns=struct.pack(">H", 0))
    cases["grey16-trns.png"] = raw_png(grey << 8, 0, 16, trns=struct.pack(">H", first << 8))
    r, g, b = (int(v) for v in rgb[0, 0])
    cases["rgb8-trns.png"] = raw_png(rgb, 2, 8, trns=struct.pack(">HHH", r, g, b))
    cases["rgb16-trns.png"] = raw_png(
        rgb << 8, 2, 16, trns=struct.pack(">HHH", r << 8, g << 8, b << 8)
    )
    pal = crop.quantize(16)
    idx = np.asarray(pal, dtype=np.uint32)[..., None]
    plte = bytes(pal.getpalette()[:48])
    cases["palette-alpha.png"] = raw_png(idx, 3, 4, plte=plte, trns=bytes(range(0, 160, 20)))
    cases["palette-short.png"] = raw_png(idx, 3, 4, plte=plte[:24])  # indices past the palette
    return cases


def orientation_cases(photo: Image.Image) -> dict[str, bytes]:
    jpeg = _encode(photo, "JPEG", quality=92)
    png = _encode(photo, "PNG")
    cases = {}
    for k in range(1, 9):
        cases[f"orient{k}.jpg"] = _encode(photo, "JPEG", quality=92, exif=_exif(k))
        cases[f"orient{k}.png"] = _encode(photo, "PNG", exif=_exif(k))
        cases[f"orient{k}.webp"] = _encode(photo, "WEBP", lossless=True, exif=_exif(k))
    six = _tiff([(0x0112, 3, 1, _short(6))])
    cases["exif-double-prefix.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, b"Exif\0\0Exif\0\0" + six)])
    cases["exif-long-be.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 4, 1, struct.pack(">I", 6))]))]
    )
    cases["exif-long-le.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 4, 1, struct.pack("<I", 8))], False))]
    )
    cases["exif-sshort.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 8, 1, _short(6))]))]
    )
    cases["exif-byte.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 1, 1, b"\x06\0\0\0")]))]
    )
    rational = _tiff([(0x0112, 5, 1, struct.pack(">I", 26))], extra=struct.pack(">II", 12, 2))
    cases["exif-rational.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, b"Exif\0\0" + rational)])
    cases["exif-count2.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 3, 2, b"\x00\x06\x00\x03")]))]
    )
    # Values stored at an offset (26 = TIFF header 8 + entry count 2 + one entry 12 + 4).
    at = struct.pack(">I", 26)
    for name, typ, count, data in [
        ("exif-short-count3", 3, 3, struct.pack(">HHH", 8, 3, 3)),
        ("exif-long-count2", 4, 2, struct.pack(">II", 5, 3)),
        ("exif-double", 12, 1, struct.pack(">d", 6.0)),
        ("exif-rational-count2", 5, 2, struct.pack(">IIII", 7, 1, 3, 1)),
        ("exif-rational-half", 5, 1, struct.pack(">II", 13, 2)),
    ]:
        block = b"Exif\0\0" + _tiff([(0x0112, typ, count, at)], extra=data)
        cases[f"{name}.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, block)])
    for name, typ, count, field in [
        ("exif-count0", 3, 0, _short(6)),
        ("exif-float", 11, 1, struct.pack(">f", 6.0)),
        ("exif-undefined", 7, 1, b"\x06\0\0\0"),
        ("exif-offset-outside", 3, 3, struct.pack(">I", 9999)),
    ]:
        block = b"Exif\0\0" + _tiff([(0x0112, typ, count, field)])
        cases[f"{name}.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, block)])
    cases["exif-duplicate.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 3, 1, _short(1)), (0x0112, 3, 1, _short(6))]))]
    )
    cases["exif-two-segments.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + six), (0xE1, b"Exif\0\0" + _tiff([(0x0112, 3, 1, _short(3))]))]
    )
    outside = b"Exif\0\0MM\x00\x2a" + struct.pack(">I", 200)  # IFD offset past the block
    cases["exif-ifd-outside.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, outside), (0xFE, six)])
    cases["exif-garbage.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, b"Exif\0\0GARBAGE!!!!")])
    xmp = b"http://ns.adobe.com/xap/1.0/\x00" + _xmp(6)
    cases["xmp-only.jpg"] = _jpeg_with_segments(jpeg, [(0xE1, xmp)])
    cases["xmp-and-exif.jpg"] = _jpeg_with_segments(
        jpeg, [(0xE1, b"Exif\0\0" + _tiff([(0x0112, 3, 1, _short(3))])), (0xE1, xmp)]
    )
    cases["xmp-itxt.png"] = _png_insert(
        png, _chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + _xmp(8))
    )
    cases["xmp-text.png"] = _png_insert(png, _chunk(b"tEXt", b"XML:com.adobe.xmp\0" + _xmp(5)))
    cases["xmp.webp"] = _encode(photo, "WEBP", lossless=True, xmp=_xmp(7))
    cases["xmp-itxt-bad-utf8.png"] = _png_insert(
        png, _chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + _xmp(6) + b"\xff")
    )
    cases["xmp-itxt-compressed.png"] = _png_insert(
        png, _chunk(b"iTXt", b"XML:com.adobe.xmp\0\1\0\0\0" + zlib.compress(_xmp(3)))
    )
    cases["exif-prefixed.png"] = _png_insert(png, _chunk(b"eXIf", b"Exif\0\0" + six))
    cases["exif-two-chunks.png"] = _png_insert(
        png, _chunk(b"eXIf", six) + _chunk(b"eXIf", _tiff([(0x0112, 3, 1, _short(3))]))
    )
    hexdump = six.hex()
    profile = f"\nexif\n{len(six):8d}\n" + "\n".join(
        hexdump[i : i + 72] for i in range(0, len(hexdump), 72)
    )
    cases["raw-profile-text.png"] = _png_insert(
        png, _chunk(b"tEXt", b"Raw profile type exif\0" + profile.encode())
    )
    cases["raw-profile-ztxt.png"] = _png_insert(
        png, _chunk(b"zTXt", b"Raw profile type exif\0\0" + zlib.compress(profile.encode()))
    )
    plain_webp = _encode(photo, "WEBP", lossless=True)
    exif_chunk = b"EXIF" + struct.pack("<I", len(six)) + six
    no_flag = plain_webp + exif_chunk
    cases["webp-exif-no-flag.webp"] = (
        no_flag[:4] + struct.pack("<I", len(no_flag) - 8) + no_flag[8:]
    )
    two = (
        _encode(photo, "WEBP", lossless=True, exif=six)
        + b"EXIF"
        + struct.pack("<I", len(six))
        + _tiff([(0x0112, 3, 1, _short(3))])
    )
    cases["webp-two-exif.webp"] = two[:4] + struct.pack("<I", len(two) - 8) + two[8:]
    return cases


def damaged_cases(photo: Image.Image) -> dict[str, bytes]:
    jpeg = _encode(photo, "JPEG", quality=92)
    png = _encode(photo, "PNG")
    webp = _encode(photo, "WEBP", quality=90)
    cases = {}
    cases["jpeg-fill-bytes.jpg"] = jpeg[:2] + b"\xff\xff\xff" + jpeg[2:]
    app0_end = 4 + struct.unpack(">H", jpeg[4:6])[0]  # SOI, then the JFIF segment
    cases["jpeg-junk-between.jpg"] = jpeg[:app0_end] + b"\x00\x00" + jpeg[app0_end:]
    cases["jpeg-no-eoi.jpg"] = jpeg[:-2]
    cases["jpeg-junk-after-eoi.jpg"] = jpeg + b"trailing junk" * 4
    cases["truncated.jpg"] = jpeg[: len(jpeg) // 2]
    cases["truncated.png"] = png[: len(png) // 2]
    cases["truncated.webp"] = webp[: len(webp) // 2]
    cases["png-no-iend.png"] = png[:-12]
    bad_crc = bytearray(png)
    bad_crc[29] ^= 0xFF  # IHDR checksum
    cases["png-bad-crc.png"] = bytes(bad_crc)
    ihdr = png[8:33]
    hostile = struct.pack(">I", 0xFFFFFFF0) + b"eXIf" + _tiff([(0x0112, 3, 1, _short(6))])
    cases["hostile-exif-size.png"] = b"\x89PNG\r\n\x1a\n" + ihdr + hostile
    # PNG metadata limits (Pillow: 1 MiB per compressed chunk, 64 MiB of text in total).
    mib = 1024 * 1024
    bomb = zlib.compress(b"\0" * 64 * mib, 9)
    before_iend = lambda extra: png[:-12] + extra + png[-12:]  # noqa: E731
    cases["ztxt-bomb.png"] = _png_insert(png, _chunk(b"zTXt", b"Comment\0\0" + bomb))
    cases["ztxt-bomb-after-idat.png"] = before_iend(_chunk(b"zTXt", b"Comment\0\0" + bomb))
    cases["itxt-bomb.png"] = _png_insert(png, _chunk(b"iTXt", b"Comment\0\1\0\0\0" + bomb))
    cases["iccp-bomb.png"] = _png_insert(png, _chunk(b"iCCP", b"icc\0\0" + bomb))
    cases["ztxt-1mib.png"] = _png_insert(
        png, _chunk(b"zTXt", b"Comment\0\0" + zlib.compress(b"a" * mib))
    )
    cases["ztxt-65-mib.png"] = _png_insert(
        png,
        b"".join(_chunk(b"zTXt", b"C%d\0\0" % i + zlib.compress(b"a" * mib)) for i in range(65)),
    )
    cases["ztxt-bad-method.png"] = _png_insert(
        png, _chunk(b"zTXt", b"Comment\0\1" + zlib.compress(b"x"))
    )
    cases["ztxt-corrupt.png"] = _png_insert(png, _chunk(b"zTXt", b"Comment\0\0garbage"))
    cases["iccp-bad-method.png"] = _png_insert(
        png, _chunk(b"iCCP", b"icc\0\1" + zlib.compress(b"x"))
    )
    cases["too-elongated.png"] = _encode(Image.new("RGB", (2600, 200), "white"), "PNG")
    cases["too-large.png"] = _encode(Image.new("L", (6000, 4500)), "PNG")
    cases["gif.gif"] = _encode(photo, "GIF")
    cases["text.jpg"] = b"this is not an image\n" * 10
    return cases


def format_cases(photo: Image.Image, icc_profile: bytes | None) -> dict[str, bytes]:
    w, h = photo.size
    alpha = Image.linear_gradient("L").resize((w, h)).rotate(90)
    rgba = photo.copy()
    rgba.putalpha(alpha)
    la = photo.convert("L")
    la.putalpha(alpha)
    # Taller than about 2:1, Pillow 13 resizes vertically first (web/preprocess.js).
    tall = photo.rotate(90, expand=True).resize((300, 1000), Image.Resampling.LANCZOS)
    cases = {
        "tall.png": _encode(tall, "PNG"),
        "tall.jpg": _encode(tall, "JPEG", quality=92),
        "tall-rotated.jpg": _encode(  # stored wide, upright after EXIF rotation
            tall.rotate(90, expand=True), "JPEG", quality=92, exif=_exif(6)
        ),
        "rgba.png": _encode(rgba, "PNG"),
        "rgba.webp": _encode(rgba, "WEBP", lossless=True),
        "rgba-lossy.webp": _encode(rgba, "WEBP", quality=90),
        "la.png": _encode(la, "PNG"),
        "palette-transparent.png": _encode(photo.quantize(64), "PNG", transparency=0),
        "grey.jpg": _encode(photo.convert("L"), "JPEG", quality=92),
        "cmyk.jpg": _encode(photo.convert("CMYK"), "JPEG", quality=92),
        "progressive.jpg": _encode(photo, "JPEG", quality=92, progressive=True),
        "subsampling-444.jpg": _encode(photo, "JPEG", quality=92, subsampling=0),
        "lossy.webp": _encode(photo, "WEBP", quality=90),
        "mpo.jpg": _encode(photo, "MPO", save_all=True, append_images=[photo.rotate(180)]),
        "animated.webp": _encode(photo, "WEBP", save_all=True, append_images=[photo.rotate(180)]),
        "apng-default-image.png": _encode(
            photo,
            "PNG",
            save_all=True,
            default_image=True,
            append_images=[photo.rotate(180), photo.rotate(90)],
        ),  # fmt: skip
    }
    if icc_profile:
        cases["display-p3.jpg"] = _encode(photo, "JPEG", quality=92, icc_profile=icc_profile)
    return cases


# Files found by differential fuzzing (one per class of difference it found), kept as they
# are: some cannot be made with Pillow (4:4:0, arithmetic-coded and lossless JPEG).
FOUND_BY_FUZZING = Path(__file__).resolve().parents[2] / "tests" / "data" / "web-edge"


def edge_cases(photo: Image.Image, icc_profile: bytes | None = None) -> dict[str, bytes]:
    """name -> file bytes. The extension is only for humans: the page sniffs the bytes."""
    found = {f"fuzz-{f.name}": f.read_bytes() for f in sorted(FOUND_BY_FUZZING.iterdir())}
    return {**orientation_cases(photo), **format_cases(photo, icc_profile),
            **png_matrix(photo), **damaged_cases(photo), **found}  # fmt: skip
