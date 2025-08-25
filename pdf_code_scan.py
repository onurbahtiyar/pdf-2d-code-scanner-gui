#!/usr/bin/env python3
"""
PDF Code Scanner (Console)
-------------------------
Tek satır komutla PDF içindeki QR ve diğer 2D barkodları tarar.

Kullanım:
    python pdf_code_scan.py dosya.pdf

Varsayılanlar:
- Mod: auto  -> opencv-contrib yüklüyse QR+DataMatrix+Aztec+PDF417, yoksa QR
- DPI: 300 (hızlı) -> 500 (gerekirse)
- Derin tarama: kapalı
- Agresif (gömülü görseller): kapalı

Opsiyonlar için: python pdf_code_scan.py -h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode  # optional
except Exception:
    dmtx_decode = None


# ---------------- PDF -> Image yardımcıları ----------------

def render_page_to_bgr(pdf_path: str, page_index: int, dpi: int) -> np.ndarray:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def extract_embedded_images_to_bgr(pdf_path: str, page_index: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        for xref, *_ in page.get_images(full=True):
            try:
                base = doc.extract_image(xref)
                nparr = np.frombuffer(base["image"], np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    out.append(bgr)
            except Exception:
                pass
    return out


# ---------------- Görüntü işleme / deteksiyon ----------------

def has_barcode_detector() -> bool:
    return hasattr(cv2, "barcode_BarcodeDetector")


def generate_variants(img_bgr: np.ndarray) -> Iterable[np.ndarray]:
    # Minimum varyant seti: orijinal, küçük büyütme, Otsu, adaptive, ters
    yield img_bgr
    up = cv2.resize(img_bgr, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    yield up
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    adap = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)
    yield cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR)
    yield cv2.bitwise_not(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))


def detect_qr_only_fast(img_bgr: np.ndarray) -> List[Tuple[str, str]]:
    det = cv2.QRCodeDetector()
    out: List[Tuple[str, str]] = []

    # Multi
    try:
        ok, info, _, _ = det.detectAndDecodeMulti(img_bgr)  # type: ignore
        if ok and info:
            out.extend([("QR", t) for t in info if t])
    except TypeError:
        try:
            info, _, _ = det.detectAndDecodeMulti(img_bgr)  # type: ignore
            out.extend([("QR", t) for t in info if t])
        except Exception:
            pass

    # Single
    if not out:
        t, _, _ = det.detectAndDecode(img_bgr)
        if t:
            out.append(("QR", t))

    # 180° dene
    if not out:
        img_rot = cv2.rotate(img_bgr, cv2.ROTATE_180)
        try:
            ok, info, _, _ = det.detectAndDecodeMulti(img_rot)  # type: ignore
            if ok and info:
                out.extend([("QR", t) for t in info if t])
        except TypeError:
            try:
                info, _, _ = det.detectAndDecodeMulti(img_rot)  # type: ignore
                out.extend([("QR", t) for t in info if t])
            except Exception:
                pass
        if not out:
            t, _, _ = det.detectAndDecode(img_rot)
            if t:
                out.append(("QR", t))

    # benzersizleştir
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for ty, tx in out:
        key = (ty, tx)
        if key not in seen:
            seen.add(key)
            uniq.append((ty, tx))
    return uniq


def detect_qr_only_deep(img_bgr: np.ndarray) -> List[Tuple[str, str]]:
    det = cv2.QRCodeDetector()
    found: List[Tuple[str, str]] = []
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

    def _try(bgr: np.ndarray) -> List[str]:
        out: List[str] = []
        try:
            ok, info, _, _ = det.detectAndDecodeMulti(bgr)  # type: ignore
            if ok and info:
                out.extend([t for t in info if t])
        except TypeError:
            try:
                info, _, _ = det.detectAndDecodeMulti(bgr)  # type: ignore
                out.extend([t for t in info if t])
            except Exception:
                pass
        if not out:
            t, _, _ = det.detectAndDecode(bgr)
            if t:
                out.append(t)
        return out

    for variant in generate_variants(img_bgr):
        for r in rotations:
            cand = variant if r is None else cv2.rotate(variant, r)
            hits = _try(cand)
            if hits:
                found.extend([("QR", t) for t in hits])
                return found

    return found


def detect_all2d_fast(img_bgr: np.ndarray) -> List[Tuple[str, str]]:
    """
    QR + DataMatrix + Aztec + PDF417 (opencv-contrib gerekir).
    pylibdmtx varsa DataMatrix'e ikinci şans.
    """
    out: List[Tuple[str, str]] = []

    # 1) BarcodeDetector (varsa)
    if has_barcode_detector():
        try:
            bd = cv2.barcode_BarcodeDetector()
            ok, infos, types, _ = bd.detectAndDecode(img_bgr)  # type: ignore
            if ok and infos:
                for t, ty in zip(infos, types):
                    if t:
                        out.append((ty or "UNKNOWN", t))
        except Exception:
            pass
        if not out:
            img_rot = cv2.rotate(img_bgr, cv2.ROTATE_180)
            try:
                ok, infos, types, _ = bd.detectAndDecode(img_rot)  # type: ignore
                if ok and infos:
                    for t, ty in zip(infos, types):
                        if t:
                            out.append((ty or "UNKNOWN", t))
            except Exception:
                pass

    # 2) QR detector ayrıca
    if not out:
        out.extend(detect_qr_only_fast(img_bgr))

    # 3) pylibdmtx ile DataMatrix
    if not out and dmtx_decode is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hits = dmtx_decode(gray)
            out.extend([("DATAMATRIX", h.data.decode("utf-8", errors="ignore"))
                        for h in hits if getattr(h, "data", None)])
        except Exception:
            pass

    # benzersiz
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for ty, tx in out:
        key = (ty, tx)
        if key not in seen:
            seen.add(key)
            uniq.append((ty, tx))
    return uniq


def detect_all2d_deep(img_bgr: np.ndarray) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

    for variant in generate_variants(img_bgr):
        for r in rotations:
            cand = variant if r is None else cv2.rotate(variant, r)

            # BarcodeDetector
            if has_barcode_detector():
                try:
                    bd = cv2.barcode_BarcodeDetector()
                    ok, infos, types, _ = bd.detectAndDecode(cand)  # type: ignore
                    if ok and infos:
                        for t, ty in zip(infos, types):
                            if t:
                                found.append((ty or "UNKNOWN", t))
                        if found:
                            return found
                except Exception:
                    pass

            # QR
            qr = detect_qr_only_fast(cand)
            if qr:
                found.extend(qr)
                return found

            # pylibdmtx DataMatrix
            if dmtx_decode is not None:
                try:
                    gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
                    hits = dmtx_decode(gray)
                    if hits:
                        found.extend([("DATAMATRIX", h.data.decode("utf-8", errors="ignore"))
                                      for h in hits if getattr(h, "data", None)])
                        return found
                except Exception:
                    pass

    return found


# ---------------- PDF tarama ----------------

def scan_pdf(
    pdf_path: Path,
    mode: str = "auto",
    dpi_low: int = 300,
    dpi_high: int = 500,
    deep: bool = False,
    aggressive: bool = False,
) -> List[Tuple[int, str, str]]:
    """
    return: [(page_no, type, text), ...]
    """
    results: List[Tuple[int, str, str]] = []

    # auto -> all2d eğer BarcodeDetector varsa, yoksa qr
    effective_mode = mode
    if mode == "auto":
        effective_mode = "all" if has_barcode_detector() else "qr"

    with fitz.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            # (opsiyonel) gömülü görseller
            if aggressive:
                for img in extract_embedded_images_to_bgr(str(pdf_path), i):
                    if effective_mode == "all":
                        hits = detect_all2d_fast(img)
                        if not hits and deep:
                            hits = detect_all2d_deep(img)
                    else:
                        hits = detect_qr_only_fast(img)
                        if not hits and deep:
                            hits = detect_qr_only_deep(img)
                    for ty, tx in hits:
                        results.append((i + 1, ty, tx))
                if results:
                    # Aynı sayfadan en az bir sonuç alındıysa raster'a geçmeden devam
                    continue

            # düşük DPI
            bgr = render_page_to_bgr(str(pdf_path), i, dpi_low)
            hits = detect_all2d_fast(bgr) if effective_mode == "all" else detect_qr_only_fast(bgr)

            # yüksek DPI / derin
            if not hits and dpi_high and dpi_high != dpi_low:
                bgr_hi = render_page_to_bgr(str(pdf_path), i, dpi_high)
                if effective_mode == "all":
                    hits = detect_all2d_fast(bgr_hi)
                    if not hits and deep:
                        hits = detect_all2d_deep(bgr_hi)
                else:
                    hits = detect_qr_only_fast(bgr_hi)
                    if not hits and deep:
                        hits = detect_qr_only_deep(bgr_hi)

            elif not hits and deep:
                if effective_mode == "all":
                    hits = detect_all2d_deep(bgr)
                else:
                    hits = detect_qr_only_deep(bgr)

            for ty, tx in hits:
                results.append((i + 1, ty, tx))

    return results


# ---------------- CLI ----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDF içindeki QR/2D barkodları tarar ve içeriğini yazdırır.")
    p.add_argument("pdf_path", type=str, help="PDF dosya yolu")
    p.add_argument("--mode", choices=["auto", "qr", "all"], default="auto",
                   help="auto: varsa tüm 2D (opencv-contrib), yoksa QR")
    p.add_argument("--dpi-low", type=int, default=300, help="Hızlı tur DPI (varsayılan 300)")
    p.add_argument("--dpi-high", type=int, default=500, help="Gerekirse yükseltilecek DPI (0 = kapalı)")
    p.add_argument("--deep", action="store_true", help="Gerekirse derin ön-işleme dene")
    p.add_argument("--aggressive", action="store_true", help="Gömülü görselleri de tara (yavaş ama güçlü)")
    p.add_argument("--json", action="store_true", help="JSON çıktısı üret (satır satır yerine tek JSON)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Hata: '{pdf_path}' bulunamadı.", file=sys.stderr)
        sys.exit(1)

    try:
        hits = scan_pdf(
            pdf_path=pdf_path,
            mode=args.mode,
            dpi_low=args.dpi_low,
            dpi_high=args.dpi_high,
            deep=args.deep,
            aggressive=args.aggressive,
        )
    except Exception as ex:
        print(f"Hata: {ex}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        import json as _json
        obj = [{"page": p, "type": ty, "text": tx} for (p, ty, tx) in hits]
        print(_json.dumps(obj, ensure_ascii=False, indent=2))
        sys.exit(0)

    if not hits:
        print("Kod bulunamadı.")
        sys.exit(0)

    for page_no, ty, text in hits:
        print(f"Sayfa {page_no} [{ty}]: {text}")


if __name__ == "__main__":
    main()
