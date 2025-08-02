// Analysis Module for Parkinson Voice Analysis
class AnalysisManager {
  constructor() {
    this.currentResults = null;
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Export functionality
    document.addEventListener("click", (e) => {
      if (e.target.matches("#exportResultsBtn")) {
        this.exportResults();
      }
      if (e.target.matches("#downloadReportBtn")) {
        this.downloadReport();
      }
      if (e.target.matches("#shareResultsBtn")) {
        this.shareResults();
      }
    });

    // Feature item interactions
    document.addEventListener("click", (e) => {
      if (e.target.closest(".feature-item")) {
        this.showFeatureDetails(e.target.closest(".feature-item"));
      }
    });

    // Chart interactions
    this.setupChartInteractions();
  }

  setupChartInteractions() {
    // Add event listeners for chart interactions when they're created
    document.addEventListener("plotly_click", (data) => {
      console.log("Chart clicked:", data);
      this.handleChartClick(data);
    });

    document.addEventListener("plotly_hover", (data) => {
      this.handleChartHover(data);
    });
  }

  setResults(results) {
    console.log("setResults called with:", results);
    this.currentResults = results;
    this.addExportButtons();
  }

  addExportButtons() {
    // Add export buttons to results section if not already present
    const resultsSection = document.getElementById("resultsSection");
    if (resultsSection && !document.getElementById("exportButtons")) {
      const exportButtonsHTML = `
                <div id="exportButtons" class="text-center mb-4">
                    <div class="btn-group" role="group">
                        <button type="button" id="downloadReportBtn" class="btn btn-primary">
                            <i class="fas fa-download me-2"></i>Raporu İndir
                        </button>
                        <button type="button" id="exportResultsBtn" class="btn btn-success">
                            <i class="fas fa-file-export me-2"></i>Verileri Dışa Aktar
                        </button>
                        <button type="button" id="shareResultsBtn" class="btn btn-info">
                            <i class="fas fa-share-alt me-2"></i>Paylaş
                        </button>
                    </div>
                </div>
            `;

      const firstCard = resultsSection.querySelector(".card");
      if (firstCard) {
        firstCard.insertAdjacentHTML("beforebegin", exportButtonsHTML);
      }
    }
  }

  exportResults() {
    if (!this.currentResults) {
      ParkinsonsApp.showToast("Dışa aktarılacak sonuç bulunamadı.", "warning");
      return;
    }

    try {
      const exportData = {
        timestamp: new Date().toISOString(),
        prediction: this.currentResults.prediction,
        features: this.currentResults.features,
        healthy_averages: this.currentResults.healthy_averages,
        parkinson_averages: this.currentResults.parkinson_averages,
        metadata: {
          version: "1.0",
          model: "Parkinson Voice Analysis",
          disclaimer:
            "Bu analiz tıbbi tanı değildir. Mutlaka bir doktora danışınız.",
        },
      };

      const filename = `parkinson_analysis_${
        new Date().toISOString().split("T")[0]
      }.json`;
      ParkinsonsApp.downloadAsJSON(exportData, filename);

      ParkinsonsApp.showToast(
        "Analiz sonuçları başarıyla dışa aktarıldı!",
        "success"
      );
    } catch (error) {
      console.error("Export error:", error);
      ParkinsonsApp.showToast("Dışa aktarma sırasında hata oluştu.", "danger");
    }
  }

  downloadReport() {
    if (!this.currentResults) {
      ParkinsonsApp.showToast("İndirilecek rapor bulunamadı.", "warning");
      return;
    }

    const removeLoading = ParkinsonsApp.addLoadingToButton(
      document.getElementById("downloadReportBtn"),
      "Rapor Hazırlanıyor..."
    );

    try {
      const reportHTML = this.generateHTMLReport();
      this.downloadHTMLReport(reportHTML);

      ParkinsonsApp.showToast("Rapor başarıyla indirildi!", "success");
    } catch (error) {
      console.error("Report generation error:", error);
      ParkinsonsApp.showToast(
        "Rapor oluşturma sırasında hata oluştu.",
        "danger"
      );
    } finally {
      removeLoading();
    }
  }

  generateHTMLReport() {
    const prediction = this.currentResults.prediction;
    const features = this.currentResults.features;
    const timestamp = new Date().toLocaleString("tr-TR");

    const isParkinson = prediction.prediction === 1;
    const probability = (prediction.probability * 100).toFixed(1);
    const confidence = (prediction.confidence * 100).toFixed(1);

    const reportHTML = `
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parkinson Ses Analizi Raporu</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .result-box { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .feature-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .feature-table th, .feature-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .feature-table th { background-color: #007bff; color: white; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .success { color: #28a745; } .danger { color: #dc3545; } .warning-text { color: #856404; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Parkinson Hastalığı Ses Analizi Raporu</h1>
        <p>Tarih: ${timestamp}</p>
    </div>
    
    <div class="result-box">
        <h2>Analiz Sonucu</h2>
        <p><strong>Durum:</strong> <span class="${
          isParkinson ? "danger" : "success"
        }">${isParkinson ? "Risk Tespit Edildi" : "Normal Bulgular"}</span></p>
        <p><strong>Olasılık:</strong> %${probability}</p>
        <p><strong>Güven Seviyesi:</strong> %${confidence}</p>
        <p><strong>Risk Seviyesi:</strong> ${
          prediction.risk_level === "High"
            ? "Yüksek"
            : prediction.risk_level === "Medium"
            ? "Orta"
            : "Düşük"
        }</p>
    </div>
    
    <div class="warning">
        <h3>⚠️ Önemli Uyarı</h3>
        <p class="warning-text">Bu analiz tıbbi tanı değildir ve kesinlikle bir doktora danışılmalıdır. Sonuçlar sadece araştırma amaçlıdır.</p>
    </div>
    
    <h2>Detaylı Ses Özellikleri</h2>
    <table class="feature-table">
        <thead>
            <tr><th>Özellik</th><th>Değer</th><th>Açıklama</th></tr>
        </thead>
        <tbody>
            ${Object.entries(features)
              .map(
                ([key, value]) => `
                <tr>
                    <td>${key}</td>
                    <td>${ParkinsonsApp.formatFeatureValue(value, key)}</td>
                    <td>${this.getFeatureDescription(key)}</td>
                </tr>
            `
              )
              .join("")}
        </tbody>
    </table>
    
    <div style="margin-top: 40px; font-size: 12px; color: #6c757d; text-align: center;">
        <p>Bu rapor Parkinson Ses Analizi Sistemi tarafından otomatik olarak oluşturulmuştur.</p>
        <p>Sistem, gelişmiş makine öğrenmesi ve derin öğrenme teknolojilerini kullanmaktadır.</p>
    </div>
</body>
</html>`;

    return reportHTML;
  }

  downloadHTMLReport(htmlContent) {
    const blob = new Blob([htmlContent], { type: "text/html" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `parkinson_analiz_raporu_${
      new Date().toISOString().split("T")[0]
    }.html`;
    link.click();

    URL.revokeObjectURL(url);
  }

  shareResults() {
    if (!this.currentResults) {
      ParkinsonsApp.showToast("Paylaşılacak sonuç bulunamadı.", "warning");
      return;
    }

    const prediction = this.currentResults.prediction;
    const isParkinson = prediction.prediction === 1;
    const probability = (prediction.probability * 100).toFixed(1);

    const shareText = `Parkinson Ses Analizi Sonucu:
        ${isParkinson ? "Risk Tespit Edildi" : "Normal Bulgular"}
Olasılık: %${probability}
Risk Seviyesi: ${prediction.risk_level}

⚠️ Bu analiz tıbbi tanı değildir. Mutlaka bir doktora danışınız.

#ParkinsonAnalizi #SesAnalizi #SağlıkTeknolojisi`;

    if (navigator.share) {
      navigator
        .share({
          title: "Parkinson Ses Analizi Sonucu",
          text: shareText,
          url: window.location.href,
        })
        .then(() => {
          ParkinsonsApp.showToast("Sonuçlar başarıyla paylaşıldı!", "success");
        })
        .catch((error) => {
          console.error("Share error:", error);
          this.fallbackShare(shareText);
        });
    } else {
      this.fallbackShare(shareText);
    }
  }

  fallbackShare(text) {
    ParkinsonsApp.copyToClipboard(text);
    ParkinsonsApp.showToast("Sonuçlar panoya kopyalandı!", "success");
  }

  showFeatureDetails(featureElement) {
    const featureName =
      featureElement.querySelector(".feature-name").textContent;
    const featureValue =
      featureElement.querySelector(".feature-value").textContent;

    const modalHTML = `
            <div class="modal fade" id="featureDetailModal" tabindex="-1">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${featureName}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p><strong>Değer:</strong> ${featureValue}</p>
                            <p><strong>Açıklama:</strong></p>
                            <p>${this.getFeatureDescription(featureName)}</p>
                            ${this.getFeatureComparison(featureName)}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Kapat</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

    // Remove existing modal
    const existingModal = document.getElementById("featureDetailModal");
    if (existingModal) {
      existingModal.remove();
    }

    // Add new modal
    document.body.insertAdjacentHTML("beforeend", modalHTML);

    // Show modal
    const modal = new bootstrap.Modal(
      document.getElementById("featureDetailModal")
    );
    modal.show();
  }

  getFeatureDescription(featureName) {
    const descriptions = {
      "MDVP:Fo(Hz)":
        "Ortalama temel ses frekansı (Hz cinsinden). Ses tellerinin titreşim hızını gösterir.",
      "MDVP:Fhi(Hz)": "Maksimum temel ses frekansı. En yüksek ses perdesi.",
      "MDVP:Flo(Hz)": "Minimum temel ses frekansı. En düşük ses perdesi.",
      "MDVP:Jitter(%)":
        "Frekans değişkenliği yüzdesi. Ses titremelerini ölçer.",
      "MDVP:Jitter(Abs)":
        "Mutlak frekans değişkenliği. Mikrosaniye cinsinden jitter.",
      "MDVP:RAP": "Rölatif ortalama pertürbasyon. Frekans varyasyonu ölçüsü.",
      "MDVP:PPQ": "Beş nokta periyot pertürbasyon katsayısı.",
      "Jitter:DDP": "Jitter:RAP değerinin üç katı.",
      "MDVP:Shimmer":
        "Genlik değişkenliği ölçüsü. Ses gücü titremelerini gösterir.",
      "MDVP:Shimmer(dB)": "Desibel cinsinden genlik değişkenliği.",
      "Shimmer:APQ3": "Üç nokta genlik pertürbasyon katsayısı.",
      "Shimmer:APQ5": "Beş nokta genlik pertürbasyon katsayısı.",
      "MDVP:APQ": "On bir nokta genlik pertürbasyon katsayısı.",
      "Shimmer:DDA": "Shimmer:APQ3 değerinin üç katı.",
      NHR: "Gürültü-harmonik oranı. Ses kalitesini ölçer.",
      HNR: "Harmonik-gürültü oranı. Ses berraklığının göstergesi.",
      RPDE: "Tekrarlama periyodu yoğunluk entropisi. Ses düzensizliğini ölçer.",
      DFA: "Detrend dalgalanma analizi. Ses sinyalinin karmaşıklığını ölçer.",
      spread1: "Fundamental frekans değişim ölçüsü.",
      spread2: "Fundamental frekans değişim ölçüsü (ikinci tür).",
      D2: "Korelasyon boyutu. Nonlineer dinamik karmaşıklık ölçüsü.",
      PPE: "Pitch periyodu entropisi. Ses periyodunun düzensizliğini ölçer.",
    };

    return descriptions[featureName] || "Bu özellik için açıklama bulunmuyor.";
  }

  getFeatureComparison(featureName) {
    if (!this.currentResults) return "";

    const userValue = this.currentResults.features[featureName];
    const healthyAvg = this.currentResults.healthy_averages[featureName];
    const parkinsonAvg = this.currentResults.parkinson_averages[featureName];

    if (
      userValue === undefined ||
      healthyAvg === undefined ||
      parkinsonAvg === undefined
    ) {
      return "";
    }

    return `
            <div class="mt-3">
                <h6>Karşılaştırma:</h6>
                <div class="row">
                    <div class="col-4">
                        <small class="text-muted">Sağlıklı Ort.</small><br>
                        <strong class="text-success">${ParkinsonsApp.formatFeatureValue(
                          healthyAvg,
                          featureName
                        )}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted">Parkinson Ort.</small><br>
                        <strong class="text-danger">${ParkinsonsApp.formatFeatureValue(
                          parkinsonAvg,
                          featureName
                        )}</strong>
                    </div>
                    <div class="col-4">
                        <small class="text-muted">Sizin Değeriniz</small><br>
                        <strong class="text-primary">${ParkinsonsApp.formatFeatureValue(
                          userValue,
                          featureName
                        )}</strong>
                    </div>
                </div>
            </div>
        `;
  }

  handleChartClick(data) {
    console.log("Chart clicked:", data);
    // Handle chart click events
  }

  handleChartHover(data) {
    // Handle chart hover events
    // Could show additional information or tooltips
  }

  // Statistical analysis methods
  calculateZScore(value, mean, std) {
    return (value - mean) / std;
  }

  getPercentile(value, referenceArray) {
    const sorted = referenceArray.sort((a, b) => a - b);
    const rank = sorted.filter((x) => x <= value).length;
    return (rank / sorted.length) * 100;
  }

  // Advanced visualization methods
  createFeatureImportanceChart(features) {
    // Could create a feature importance visualization
    // This would require additional data about feature importance
  }

  showNotification(message, type = "info") {
    // Create notification element
    const notification = document.createElement("div");
    notification.className = `alert alert-${
      type === "error" ? "danger" : type
    } alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
      top: 20px;
      right: 20px;
      z-index: 9999;
      min-width: 300px;
      max-width: 500px;
    `;

    notification.innerHTML = `
      <i class="fas fa-${
        type === "success"
          ? "check-circle"
          : type === "error"
          ? "exclamation-triangle"
          : "info-circle"
      } me-2"></i>
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);
  }

  createTrendAnalysis(historicalData) {
    // Could create trend analysis if historical data is available
    // For tracking changes over time
  }
}

// Initialize analysis manager
document.addEventListener("DOMContentLoaded", function () {
  console.log("Initializing AnalysisManager");
  window.analysisManager = new AnalysisManager();

  // Make it available to audio recorder for setting results
  console.log("Checking for audioRecorder:", window.audioRecorder);

  // Use a more robust approach to connect with audio recorder
  const connectToAudioRecorder = () => {
    if (window.audioRecorder && window.audioRecorder.displayResults) {
      console.log("Connecting to audio recorder");
      const originalDisplayResults = window.audioRecorder.displayResults;
      window.audioRecorder.displayResults = function (result) {
        console.log("Audio recorder displayResults called with:", result);
        originalDisplayResults.call(this, result);
        window.analysisManager.setResults(result);
      };
    } else {
      console.log("Audio recorder not found, retrying in 1 second");
      setTimeout(connectToAudioRecorder, 1000);
    }
  };

  connectToAudioRecorder();
});
