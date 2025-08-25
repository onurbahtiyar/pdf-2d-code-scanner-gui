from __future__ import annotations

import sys
import os
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

import fitz  # PyMuPDF
import numpy as np

# OpenCV kurulumu:
# - Sadece QR (hızlı) için: opencv-python
# - Tüm 2D (QR+DataMatrix+Aztec+PDF417) için: opencv-contrib-python
import cv2

# (Opsiyonel) DataMatrix için ek güç
try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode  # type: ignore
except Exception:
    dmtx_decode = None

from concurrent.futures import ProcessPoolExecutor, as_completed

from PyQt6.QtCore import Qt
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import qdarktheme
    _HAS_QDARKTHEME = True
except Exception:
    _HAS_QDARKTHEME = False

# -----------------------------
# Barkod/QR yardımcıları
# -----------------------------

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


def _merge_unique(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    merged: List[Tuple[str, str]] = []
    for ty, tx in items:
        key = (ty, tx)
        if key not in seen:
            seen.add(key)
            merged.append((ty, tx))
    return merged


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

    return _merge_unique(out)


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
                return _merge_unique(found)

    return _merge_unique(found)


def has_barcode_detector() -> bool:
    return hasattr(cv2, "barcode_BarcodeDetector")


def detect_all2d(img_bgr: np.ndarray) -> List[Tuple[str, str]]:
    """
    QR + DataMatrix + Aztec + PDF417 (opencv-contrib gerekir).
    Ek olarak, pylibdmtx varsa DataMatrix'e ikinci şans.
    Hız odaklı; sadece hızlı varyant + 180° dener. (Derin mod çağırırsa varyantlar artar)
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

    return _merge_unique(out)


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
                            return _merge_unique(found)
                except Exception:
                    pass

            # QR
            qr = detect_qr_only_fast(cand)
            if qr:
                found.extend(qr)
                return _merge_unique(found)

            # pylibdmtx DataMatrix
            if dmtx_decode is not None:
                try:
                    gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
                    hits = dmtx_decode(gray)
                    if hits:
                        found.extend([("DATAMATRIX", h.data.decode("utf-8", errors="ignore"))
                                      for h in hits if getattr(h, "data", None)])
                        return _merge_unique(found)
                except Exception:
                    pass

    return _merge_unique(found)


def scan_page_worker(
    pdf_path: str,
    page_index: int,
    mode: str,
    dpi_low: int,
    dpi_high: int,
    deep: bool,
    aggressive: bool
) -> List[Tuple[int, str, str]]:
    """
    Tek sayfa tarayıcı.
    return: [(page_no, type, text), ...]
    """
    results: List[Tuple[int, str, str]] = []

    # 0) Agresif: gömülü görseller
    if aggressive:
        for img in extract_embedded_images_to_bgr(pdf_path, page_index):
            if mode == "qr":
                hits = detect_qr_only_fast(img)
                if not hits and deep:
                    hits = detect_qr_only_deep(img)
            else:
                hits = detect_all2d(img)
                if not hits and deep:
                    hits = detect_all2d_deep(img)
            for ty, tx in hits:
                results.append((page_index + 1, ty, tx))
        if results:
            return results

    # 1) Düşük DPI hızlı
    bgr = render_page_to_bgr(pdf_path, page_index, dpi_low)
    if mode == "qr":
        hits = detect_qr_only_fast(bgr)
        if not hits and dpi_high and dpi_high != dpi_low:
            bgr_hi = render_page_to_bgr(pdf_path, page_index, dpi_high)
            hits = detect_qr_only_fast(bgr_hi)
            if not hits and deep:
                hits = detect_qr_only_deep(bgr_hi)
        elif not hits and deep:
            hits = detect_qr_only_deep(bgr)
    else:
        hits = detect_all2d(bgr)
        if not hits and dpi_high and dpi_high != dpi_low:
            bgr_hi = render_page_to_bgr(pdf_path, page_index, dpi_high)
            hits = detect_all2d(bgr_hi)
            if not hits and deep:
                hits = detect_all2d_deep(bgr_hi)
        elif not hits and deep:
            hits = detect_all2d_deep(bgr)

    for ty, tx in hits:
        results.append((page_index + 1, ty, tx))

    return results


# -----------------------------
# GUI – PyQt6
# -----------------------------

@dataclass
class ScanResult:
    file_path: str
    page_no: int
    code_type: str
    text: str


class ScanWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)  # current, total, message
    result = QtCore.pyqtSignal(object)          # ScanResult
    file_done = QtCore.pyqtSignal(str, int)     # file, count
    finished_all = QtCore.pyqtSignal()

    def __init__(self, files: List[str], mode: str, dpi_low: int, dpi_high: int,
                 deep: bool, aggressive: bool, jobs: int | None, parent=None):
        super().__init__(parent)
        self.files = files
        self.mode = mode
        self.dpi_low = dpi_low
        self.dpi_high = dpi_high
        self.deep = deep
        self.aggressive = aggressive
        self.jobs = jobs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self.files)
        for idx, file in enumerate(self.files, start=1):
            if self._cancel:
                break
            try:
                with fitz.open(file) as doc:
                    page_count = len(doc)
            except Exception as e:
                self.progress.emit(idx, total, f"Hata: {os.path.basename(file)} açılamadı: {e}")
                continue

            self.progress.emit(idx, total, f"Taranıyor: {os.path.basename(file)} ({page_count} sayfa)")

            found_count = 0
            try:
                with ProcessPoolExecutor(max_workers=self.jobs) as ex:
                    futures = [
                        ex.submit(
                            scan_page_worker,
                            file,
                            p,
                            self.mode,
                            self.dpi_low,
                            self.dpi_high,
                            self.deep,
                            self.aggressive
                        ) for p in range(page_count)
                    ]
                    for fut in as_completed(futures):
                        if self._cancel:
                            break
                        try:
                            page_hits = fut.result()
                            for page_no, ty, tx in page_hits:
                                self.result.emit(ScanResult(file, page_no, ty, tx))
                                found_count += 1
                        except Exception as e:
                            self.progress.emit(idx, total, f"Uyarı: {os.path.basename(file)} sayfa tarama hatası: {e}")
            except Exception as e:
                self.progress.emit(idx, total, f"Hata: {os.path.basename(file)} taramada sorun: {e}")

            self.file_done.emit(file, found_count)
            self.progress.emit(idx, total, f"Bitti: {os.path.basename(file)} (bulunan {found_count})")

        self.finished_all.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Kod Tarayıcı")
        self.resize(1100, 700)
        self.results: List[ScanResult] = []
        self.worker: Optional[ScanWorker] = None

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # --- Sol: PDF listesi ---
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 2)

        path_row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(str(Path("./pdfs").resolve()))
        self.btn_browse = QtWidgets.QPushButton("Klasör Seç")
        self.btn_refresh = QtWidgets.QPushButton("Yenile")
        path_row.addWidget(self.folder_edit, 1)
        path_row.addWidget(self.btn_browse)
        path_row.addWidget(self.btn_refresh)
        left.addLayout(path_row)

        self.list_pdfs = QtWidgets.QListWidget()
        self.list_pdfs.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(self.list_pdfs, 1)

        row_sel = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Tümünü Seç")
        self.btn_clear_sel = QtWidgets.QPushButton("Seçimi Temizle")
        row_sel.addWidget(self.btn_select_all)
        row_sel.addWidget(self.btn_clear_sel)
        left.addLayout(row_sel)

        # --- Sağ: seçenekler + sonuçlar ---
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 3)

        # Seçenekler
        grp_opts = QtWidgets.QGroupBox("Ayarlar")
        form = QtWidgets.QFormLayout(grp_opts)

        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["QR (Hızlı)", "Tüm 2D (QR + DataMatrix + Aztec + PDF417)"])
        self.spin_dpi_low = QtWidgets.QSpinBox()
        self.spin_dpi_low.setRange(72, 1200)
        self.spin_dpi_low.setValue(300)
        self.spin_dpi_high = QtWidgets.QSpinBox()
        self.spin_dpi_high.setRange(0, 1200)
        self.spin_dpi_high.setValue(500)
        self.chk_deep = QtWidgets.QCheckBox("Derin tarama (gerekirse)")
        self.chk_aggressive = QtWidgets.QCheckBox("Agresif (gömülü görselleri tara)")
        self.spin_jobs = QtWidgets.QSpinBox()
        self.spin_jobs.setRange(0, 64)
        self.spin_jobs.setValue(0)
        self.spin_jobs.setToolTip("0 = CPU sayısı")

        form.addRow("Mod", self.cmb_mode)
        form.addRow("DPI (Hızlı)", self.spin_dpi_low)
        form.addRow("DPI (Yüksek)", self.spin_dpi_high)
        form.addRow(self.chk_deep)
        form.addRow(self.chk_aggressive)
        form.addRow("İş Parçacığı (jobs)", self.spin_jobs)

        right.addWidget(grp_opts)

        # Butonlar
        row_btns = QtWidgets.QHBoxLayout()
        self.btn_scan_selected = QtWidgets.QPushButton("Seçilenleri Tara")
        self.btn_scan_all = QtWidgets.QPushButton("Tümünü Tara")
        self.btn_cancel = QtWidgets.QPushButton("İptal")
        self.btn_cancel.setEnabled(False)
        row_btns.addWidget(self.btn_scan_selected)
        row_btns.addWidget(self.btn_scan_all)
        row_btns.addWidget(self.btn_cancel)
        right.addLayout(row_btns)

        # Sonuç tablosu
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Dosya", "Sayfa", "Tür", "İçerik"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.table, 1)

        # Dışa aktar
        row_export = QtWidgets.QHBoxLayout()
        self.btn_export_csv = QtWidgets.QPushButton("CSV Dışa Aktar")
        self.btn_export_json = QtWidgets.QPushButton("JSON Dışa Aktar")
        row_export.addWidget(self.btn_export_csv)
        row_export.addWidget(self.btn_export_json)
        right.addLayout(row_export)

        # İlerleme ve log
        self.progress = QtWidgets.QProgressBar()
        self.lbl_status = QtWidgets.QLabel("Hazır")
        right.addWidget(self.progress)
        right.addWidget(self.lbl_status)

        # Sinyaller
        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_refresh.clicked.connect(self.load_pdfs)
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        self.btn_scan_selected.clicked.connect(self.scan_selected)
        self.btn_scan_all.clicked.connect(self.scan_all)
        self.btn_cancel.clicked.connect(self.cancel_scan)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)

        # Tema
        if _HAS_QDARKTHEME:
            qdarktheme.setup_theme("dark")

        # İlk yükleme
        self.load_pdfs()

    # --- Yardımcılar ---

    def on_browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "PDF Klasörünü Seç", self.folder_edit.text())
        if path:
            self.folder_edit.setText(path)
            self.load_pdfs()

    def load_pdfs(self):
        self.list_pdfs.clear()
        folder = Path(self.folder_edit.text()).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in folder.glob("*.pdf")])
        for p in files:
            item = QtWidgets.QListWidgetItem(p.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(p))
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.list_pdfs.addItem(item)
        self.lbl_status.setText(f"{len(files)} PDF bulundu.")

    def selected_files(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list_pdfs.count()):
            item = self.list_pdfs.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked or item.isSelected():
                out.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return out

    def select_all(self):
        for i in range(self.list_pdfs.count()):
            self.list_pdfs.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def clear_selection(self):
        for i in range(self.list_pdfs.count()):
            self.list_pdfs.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.list_pdfs.item(i).setSelected(False)

    def current_mode(self) -> str:
        return "qr" if self.cmb_mode.currentIndex() == 0 else "all"

    def toggle_ui(self, busy: bool):
        for w in [self.btn_browse, self.btn_refresh, self.btn_select_all, self.btn_clear_sel,
                  self.btn_scan_selected, self.btn_scan_all, self.cmb_mode,
                  self.spin_dpi_low, self.spin_dpi_high, self.chk_deep,
                  self.chk_aggressive, self.spin_jobs]:
            w.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)

    # --- Taramalar ---

    def scan_selected(self):
        files = self.selected_files()
        if not files:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Lütfen taranacak PDF(leri) seçin.")
            return
        self.start_scan(files)

    def scan_all(self):
        files: List[str] = []
        for i in range(self.list_pdfs.count()):
            files.append(self.list_pdfs.item(i).data(QtCore.Qt.ItemDataRole.UserRole))
        if not files:
            QtWidgets.QMessageBox.information(self, "Bilgi", "pdfs klasöründe PDF bulunamadı.")
            return
        self.start_scan(files)

    def start_scan(self, files: List[str]):
        self.results.clear()
        self.table.setRowCount(0)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.lbl_status.setText("Tarama başladı…")
        self.toggle_ui(True)

        jobs = self.spin_jobs.value()
        jobs = None if jobs == 0 else jobs

        self.worker = ScanWorker(
            files=files,
            mode=self.current_mode(),
            dpi_low=self.spin_dpi_low.value(),
            dpi_high=self.spin_dpi_high.value(),
            deep=self.chk_deep.isChecked(),
            aggressive=self.chk_aggressive.isChecked(),
            jobs=jobs
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.result.connect(self.on_result)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished_all.connect(self.on_finished)
        self.worker.start()

    def cancel_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.lbl_status.setText("İptal ediliyor…")

    # --- Sinyal geri çağrıları ---

    @QtCore.pyqtSlot(int, int, str)
    def on_progress(self, current: int, total: int, message: str):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.lbl_status.setText(message)

    @QtCore.pyqtSlot(object)
    def on_result(self, res: ScanResult):
        self.results.append(res)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(os.path.basename(res.file_path)))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(res.page_no)))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(res.code_type))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(res.text))

    @QtCore.pyqtSlot(str, int)
    def on_file_done(self, file: str, found: int):
        # dosya bazlı durum istenirse burada loglanabilir
        pass

    @QtCore.pyqtSlot()
    def on_finished(self):
        self.toggle_ui(False)
        self.lbl_status.setText(f"Tamamlandı. {len(self.results)} sonuç.")

    # --- Dışa aktar ---

    def export_csv(self):
        if not self.results:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Dışa aktarılacak sonuç yok.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "CSV kaydet", "qr_results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "page", "type", "text"])
            for r in self.results:
                w.writerow([r.file_path, r.page_no, r.code_type, r.text])
        QtWidgets.QMessageBox.information(self, "Bilgi", f"CSV kaydedildi: {path}")

    def export_json(self):
        if not self.results:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Dışa aktarılacak sonuç yok.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "JSON kaydet", "qr_results.json", "JSON (*.json)")
        if not path:
            return
        data = [
            {"file": r.file_path, "page": r.page_no, "type": r.code_type, "text": r.text}
            for r in self.results
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "Bilgi", f"JSON kaydedildi: {path}")

def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    if _HAS_QDARKTHEME:
        qdarktheme.setup_theme("dark")
        return

    # Fallback: Qt Fusion dark palette
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    # Arka planlar
    p.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(42, 42, 42))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(66, 66, 66))
    # Metinler
    p.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(220, 220, 220))
    # Butonlar
    p.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    # Vurgular
    p.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(90, 90, 200))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(p)

def main():
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
