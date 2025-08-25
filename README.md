# PDF Kod Tarayıcı (PyQt6)

`pdfs/` klasörüne koyduğun PDF dosyalarındaki **QR** ve diğer **2D barkodları** (DataMatrix, Aztec, PDF417) tarayıp
içeriğini tablo olarak gösteren basit bir masaüstü uygulaması.

## Özellikler
- **Klasör tarama:** `pdfs/` içindeki PDF'leri otomatik listeler.
- **Seçimli / toplu tarama:** Seçilenleri veya tümünü tara.
- **Modlar:**
  - **QR (Hızlı):** Sadece QR kod (opencv-python yeterli).
  - **Tüm 2D:** QR + DataMatrix + Aztec + PDF417 (opencv-contrib-python gerekir).
- **Ayarlar:** DPI (Hızlı/Yüksek), Derin tarama, Gömülü görselleri tara (agresif), Paralel iş sayısı (jobs).
- **Dışa aktar:** Sonuçları **CSV** veya **JSON** olarak kaydet.
- **Tema:** Koyu tema. `qdarktheme` mevcut değilse otomatik **Fusion (dark)** temaya düşer.

## Gereksinimler
- Python **3.10 – 3.13**
- Windows / macOS / Linux
- PDF işleme için **PyMuPDF (fitz)**, görüntü işleme için **OpenCV**

## Kurulum

> Bir sanal ortam (venv) kullanman önerilir.

### 1) Sanal ortam
**Windows (PowerShell / CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Bağımlılıklar
```bash
pip install -r requirements.txt
```

> **Tüm 2D modu** (DataMatrix/Aztec/PDF417) için:
```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```
> **(Opsiyonel)** Inatçı **DataMatrix** kodları için:
```bash
pip install pylibdmtx
```

> Not: Python 3.13'te `qdarktheme` tekerleği yoksa uygulama otomatik Fusion (dark) kullanır, ekstra kurulum gerektirmez.

## Çalıştırma
```bash
python pdf_scanner_gui.py
```

- Sol panelde `pdfs/` klasöründeki PDF’ler listelenir.
- Seçim yapıp **Seçilenleri Tara** diyebilir veya **Tümünü Tara** ile hepsini işletebilirsin.
- Sağ panelde mod ve performans ayarlarını yapılandırabilirsin.
- Sonuçları tabloda görür, ister **CSV**, ister **JSON** olarak dışa aktarabilirsin.

## Klasör Yapısı
```
proje-kok/
├─ pdf_scanner_gui.py
├─ requirements.txt
├─ README.md
└─ pdfs/                  # PDF'leri buraya koy
```

> `pdfs/` klasörü yoksa uygulama oluşturur.

## Sık Karşılaşılan Sorunlar

- **`ModuleNotFoundError: No module named 'PyQt6'`**
  - Sanal ortam aktif değil veya PyQt6 kurulmadı.
  - Çözüm: `.venv`’i etkinleştir ve `pip install -r requirements.txt` çalıştır.

- **“Tüm 2D” modunda Aztec/PDF417 okunmuyor**
  - `barcode_BarcodeDetector` yok demektir; `opencv-python` kurulu.
  - Çözüm: `pip uninstall -y opencv-python && pip install opencv-contrib-python`.

- **DataMatrix bazı belgelerde hâlâ okunmuyor**
  - Çözüm: `pip install pylibdmtx` (opsiyonel ikinci deneme).

- **Koyu tema yok / `qdarktheme` yüklenemiyor**
  - Python 3.13’te `qdarktheme` yoksa otomatik Fusion (dark) kullanılır.

- **Yavaş tarama**
  - “DPI (Hızlı)” değerini **300–350**, “DPI (Yüksek)” değerini **500–600** seç.
  - **Derin tarama** ve **Agresif** seçeneklerini sadece gerektiğinde aç.
  - **İş Parçacığı (jobs)** değerini CPU çekirdeği kadar ayarla (0 = otomatik).

## Tek Dosya (EXE) Paketleme – Windows (İsteğe Bağlı)
```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name PDFKodTarayici pdf_scanner_gui.py
# Çıktı: dist/PDFKodTarayici.exe
```
> OneDrive veya boşluk/özel karakter içeren yollarda paketleme sırasında sorun yaşarsan
> projeyi kısa bir klasöre (ör. `C:\work\pdf-tarayici`) taşıyıp tekrar dene.

## Güvenlik
Uygulama tamamen yerelde çalışır; PDF’ler veya sonuçlar dış servislere gönderilmez.
