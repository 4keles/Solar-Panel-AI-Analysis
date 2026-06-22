from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class HelpWindow(QDialog):
    """Solar Panel OD Streaming dökümantasyon ve yardım penceresi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kullanım & Geliştirici Kılavuzu")
        self.resize(650, 500)
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self):
        # Pencereyi ana pencerenin üstünde bağımsız açalım
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F1117;
            }
            QTextBrowser {
                background-color: #161B27;
                border: 1px solid #1E2A3A;
                border-radius: 8px;
                padding: 12px;
                color: #E2E8F0;
                font-family: "Segoe UI", sans-serif;
            }
            QPushButton {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #CBD5E1;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #263548;
                border-color: #3B82F6;
                color: #FFFFFF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Başlık
        title_lbl = QLabel("Solar Panel OD Kullanıcı ve Geliştirici Dökümantasyonu")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_lbl.setFont(font)
        title_lbl.setStyleSheet("color: #38BDF8;")
        layout.addWidget(title_lbl)

        # Döküman İçeriği (Markdown/HTML destekli QTextBrowser)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(self._get_help_html())
        layout.addWidget(self.browser)

        # Kapat Butonu
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _get_help_html(self) -> str:
        return """
        <html>
        <head>
        <style>
            body { font-size: 13px; line-height: 1.6; color: #E2E8F0; }
            h2 { color: #38BDF8; font-size: 15px; border-bottom: 1px solid #1E2A3A; padding-bottom: 4px; margin-top: 20px; }
            h3 { color: #A78BFA; font-size: 13px; margin-top: 15px; }
            code { background-color: #1E293B; color: #34D399; padding: 2px 5px; border-radius: 4px; font-family: monospace; }
            ul { margin-left: 20px; }
            li { margin-bottom: 6px; }
            strong { color: #FFFFFF; }
            .tip { background-color: #0D2137; border-left: 4px solid #38BDF8; padding: 10px; margin: 10px 0; border-radius: 4px; }
        </style>
        </head>
        <body>
            <p>Bu arayüz, güneş panellerindeki hücre/string hatalarını (hotspot, açık devre vb.) canlı akış ve kayıtlar üzerinden gerçek zamanlı tespit etmek için tasarlanmıştır.</p>

            <h2>1. Çoklu Görüntü Kaynağı Yönetimi</h2>
            <p>Sol paneldeki <strong>KAYNAK SEÇİCİ</strong> grubundan akış türünü seçip adresini tanımlayabilirsiniz:</p>
            <ul>
                <li><strong>Webcam:</strong> Cihaza bağlı USB/tümleşik kameralar için kamera indeksini girin (örn: <code>0</code> veya <code>1</code>).</li>
                <li><strong>Yerel Video:</strong> Kayıtlı <code>.mp4</code> uçuş videolarını oynatmak için <code>...</code> butonuna basıp dosya seçin.</li>
                <li><strong>IP Kamera (HTTP):</strong> Telefon/ağ kameraları için akış adresini girin (örn: <code>http://192.168.1.50:4747/video</code>).</li>
                <li><strong>Drone Yayını (RTSP):</strong> Drone otopilotu veya RTSP sunucuları için yayın adresini girin (örn: <code>rtsp://192.168.0.50:8554/live</code>).</li>
                <li><strong>RTMP Akışı:</strong> Sunucu yayınları için RTMP adresini girin (örn: <code>rtmp://192.168.0.1/live</code>).</li>
            </ul>

            <div class="tip">
                <strong>Bağlantı Toleransı:</strong> Canlı IP/RTSP akışlarında bağlantı anlık olarak kesilirse sistem çökmez. Arayüzde <code>YENİDEN BAĞLANIYOR</code> durumu belirir, 3 saniye aralıklarla 10 kez yeniden bağlanmayı dener. Başarılı olursa akış kaldığı yerden devam eder.
            </div>

            <h2>2. Video Oynatma ve İşleme Hızı Kontrolleri</h2>
            <p>Yerel video dosyaları ile çalışırken (ve model performansını yönetirken) aşağıdaki gelişmiş kontrolleri kullanabilirsiniz:</p>
            <ul>
                <li><strong>Zaman Çizelgesi ve Seek:</strong> Video izlerken alttaki çubuğu kullanarak videonun istediğiniz anına atlayabilirsiniz. Sistem kuyruğu anında temizleyip yeni konumu gösterir.</li>
                <li><strong>Oynatma Hızı Çarpanı:</strong> Zaman çizelgesinin sağındaki açılır menüden videonun oynatma hızını (0.5x, 1x, 2x, 4x) değiştirebilir ve analiz sürecini hızlandırabilirsiniz.</li>
                <li><strong>Maksimum FPS (İşleme Hızı):</strong> Sol menüdeki Güvenilirlik Eşiği altında yer alan "İŞLEME HIZI (FPS LİMİTİ)" ile modelin saniyede yapacağı maksimum tahmin (inference) sayısını sınırlayabilirsiniz. Böylece sistem kaynaklarını (CPU/GPU) optimize edebilirsiniz (0=Sınırsız).</li>
            </ul>

            <h2>3. Termal Analiz ve Format Dönüşümleri</h2>
            <p>Termal veri setleriyle eğitilmiş modellerin (örn: model adında <code>thermal</code> veya <code>gray</code> geçenler) RGB renk sapmalarından etkilenmemesi için arayüzde <strong>GÖRÜNTÜ MODU</strong> alanı bulunur:</p>
            <ul>
                <li>Termal model seçildiğinde sistem otomatik olarak <strong>Termal Analiz</strong> modunu aktif eder.</li>
                <li><strong>Varsayılan dönüşüm Grayscale (Gri Tonlama) formatıdır</strong> (modelin beklediği standart).</li>
                <li>Görsel analizleri iyileştirmek için <strong>Inferno, Jet, Magma ve Hot</strong> renk haritalarına dinamik olarak geçiş yapabilirsiniz.</li>
                <li>Tüm dönüşümler en yüksek FPS'i sağlamak adına arka plan işlemci thread'inde verimli bir şekilde yürütülür.</li>
            </ul>

            <h2>4. Kayıt ve Klasör Yönetimi</h2>
            <p><strong>DOSYA YOLLARI YÖNETİMİ</strong> altından kayıt hedeflerini dinamik olarak değiştirebilirsiniz:</p>
            <ul>
                <li><strong>Video Kayıt Klasörü:</strong> "Kayıt Başlat" butonuna basıldığında kaydedilecek analizli videoların (.mp4) yazılacağı konum.</li>
                <li><strong>Görsel Yakalama Klasörü:</strong> "Veri Yakala" (Capture) butonuna basıldığında o ana ait ham görüntünün (.jpg) kaydedileceği konum.</li>
                <li>Seçtiğiniz klasör yollarını <strong>"Varsayılan Olarak Kaydet"</strong> butonuyla <code>streaming.yaml</code> dosyanıza kalıcı olarak yazdırabilirsiniz.</li>
            </ul>

            <h2>5. Geliştirici & Build Rehberi</h2>
            <h3>Projeyi Yerel Çalıştırma (Development)</h3>
            <p>1. Gerekli Python bağımlılıklarını kurun:</p>
            <p><code>pip install PyQt6 ultralytics opencv-python pyyaml torch numpy</code></p>
            <p>2. Uygulamayı başlatın:</p>
            <p><code>python streaming/main.py</code></p>

            <h3>Uygulamayı Dağıtıma Hazırlama (Build)</h3>
            <p>Uygulamayı bağımsız bir çalıştırılabilir dosya haline (executable) getirmek için PyInstaller kullanabilirsiniz:</p>
            <p><code>pip install pyinstaller</code></p>
            <p><code>pyinstaller --onedir --windowed --name="SolarPanelOD" --add-data "streaming/configs:configs" --add-data "streaming/src/ui/theme.qss:src/ui" streaming/main.py</code></p>
        </body>
        </html>
        """
