// Audio Recording Module for Parkinson Voice Analysis
class AudioRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.recordingTime = 0;
    this.recordingInterval = null;
    this.isRecording = false;
    this.recordedAudioBlob = null;

    this.initializeElements();
    this.setupEventListeners();
  }

  initializeElements() {
    this.recordBtn = document.getElementById("recordBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.analyzeBtn = document.getElementById("analyzeBtn");
    this.audioPlayerDiv = document.getElementById("audioPlayer");
    this.audioPlayer = this.audioPlayerDiv.querySelector("audio");
    this.audioSource = document.getElementById("audioSource");
    this.recordingStatus = document.getElementById("recordingStatus");
    this.recordingTimeDisplay = document.getElementById("recordingTime");
    this.fileUploadArea = document.getElementById("fileUploadArea");
    this.audioFileInput = document.getElementById("audioFile");
  }

  setupEventListeners() {
    // Recording controls
    this.recordBtn.addEventListener("click", () => this.startRecording());
    this.stopBtn.addEventListener("click", () => this.stopRecording());
    this.analyzeBtn.addEventListener("click", () => this.analyzeAudio());

    // File upload
    this.fileUploadArea.addEventListener("click", () =>
      this.audioFileInput.click()
    );
    this.fileUploadArea.addEventListener("dragover", (e) =>
      this.handleDragOver(e)
    );
    this.fileUploadArea.addEventListener("dragleave", (e) =>
      this.handleDragLeave(e)
    );
    this.fileUploadArea.addEventListener("drop", (e) => this.handleDrop(e));
    this.audioFileInput.addEventListener("change", (e) =>
      this.handleFileSelect(e)
    );
  }

  async startRecording() {
    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100,
        },
      });

      // Initialize MediaRecorder
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: this.getSupportedMimeType(),
      });

      this.audioChunks = [];
      this.recordingTime = 0;

      // Setup MediaRecorder events
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.createAudioBlob();
      };

      // Start recording
      this.mediaRecorder.start();
      this.isRecording = true;

      // Update UI
      this.updateRecordingUI(true);

      // Start timer
      this.startTimer();

      ParkinsonsApp.showToast(
        'Kayıt başladı. En az 3 saniye "aaa" sesi çıkarın.',
        "info"
      );
    } catch (error) {
      console.error("Recording error:", error);
      let message = "Mikrofor erişimi başarısız.";

      if (error.name === "NotAllowedError") {
        message =
          "Mikrofon izni verilmedi. Lütfen tarayıcı ayarlarından izin verin.";
      } else if (error.name === "NotFoundError") {
        message =
          "Mikrofon bulunamadı. Lütfen mikrofon bağlı olduğundan emin olun.";
      }

      ParkinsonsApp.showToast(message, "danger");
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;

      // Stop all tracks
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
      }

      // Stop timer
      this.stopTimer();

      // Update UI
      this.updateRecordingUI(false);

      ParkinsonsApp.showToast("Kayıt tamamlandı!", "success");
    }
  }

  getSupportedMimeType() {
    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/wav",
    ];

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return "audio/webm"; // Fallback
  }

  createAudioBlob() {
    const mimeType = this.mediaRecorder.mimeType;
    this.recordedAudioBlob = new Blob(this.audioChunks, { type: mimeType });

    // Create URL for audio player
    const audioURL = URL.createObjectURL(this.recordedAudioBlob);
    this.audioSource.src = audioURL;
    this.audioPlayer.load();
    this.audioPlayerDiv.style.display = "block";

    // Enable analyze button
    this.analyzeBtn.disabled = false;

    // Check recording duration
    if (this.recordingTime < 3) {
      ParkinsonsApp.showToast(
        "Kayıt süresi çok kısa. En az 3 saniye kayıt yapın.",
        "warning"
      );
    }
  }

  startTimer() {
    this.recordingTime = 0;
    this.recordingInterval = setInterval(() => {
      this.recordingTime++;
      this.recordingTimeDisplay.textContent = ParkinsonsApp.formatTime(
        this.recordingTime
      );
    }, 1000);
  }

  stopTimer() {
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
      this.recordingInterval = null;
    }
  }

  updateRecordingUI(isRecording) {
    if (isRecording) {
      this.recordBtn.disabled = true;
      this.stopBtn.disabled = false;
      this.recordingStatus.style.display = "block";
      this.analyzeBtn.disabled = true;
    } else {
      this.recordBtn.disabled = false;
      this.stopBtn.disabled = true;
      this.recordingStatus.style.display = "none";
    }
  }

  // File upload handlers
  handleDragOver(e) {
    e.preventDefault();
    this.fileUploadArea.classList.add("dragover");
  }

  handleDragLeave(e) {
    e.preventDefault();
    this.fileUploadArea.classList.remove("dragover");
  }

  handleDrop(e) {
    e.preventDefault();
    this.fileUploadArea.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  handleFile(file) {
    try {
      // Validate file
      ParkinsonsApp.validateAudioFile(file);

      // Create audio URL
      const audioURL = URL.createObjectURL(file);
      this.audioSource.src = audioURL;
      this.audioPlayer.load();
      this.audioPlayerDiv.style.display = "block";

      // Store file for analysis
      this.recordedAudioBlob = file;

      // Enable analyze button
      this.analyzeBtn.disabled = false;

      // Update UI feedback
      const uploadContent =
        this.fileUploadArea.querySelector(".upload-content");
      uploadContent.innerHTML = `
                <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                <p class="mb-1 text-success"><strong>${file.name}</strong></p>
                <small class="text-muted">Dosya yüklendi, analiz için hazır</small>
            `;

      ParkinsonsApp.showToast("Ses dosyası başarıyla yüklendi!", "success");
    } catch (error) {
      ParkinsonsApp.showToast(error.message, "danger");
    }
  }

  async analyzeAudio() {
    if (!this.recordedAudioBlob) {
      ParkinsonsApp.showToast(
        "Önce ses kaydı yapın veya dosya yükleyin.",
        "warning"
      );
      return;
    }

    const removeLoading = ParkinsonsApp.addLoadingToButton(
      this.analyzeBtn,
      "Analiz Ediliyor..."
    );

    // Show loading modal
    const loadingModal = new bootstrap.Modal(
      document.getElementById("loadingModal")
    );
    loadingModal.show();

    // Animate progress bar
    const progressBar = document.querySelector("#loadingModal .progress-bar");
    ParkinsonsApp.animateProgress(progressBar, 30, 1000);

    try {
      // Prepare form data
      const formData = new FormData();

      // Convert blob to file if necessary
      if (
        this.recordedAudioBlob instanceof Blob &&
        !(this.recordedAudioBlob instanceof File)
      ) {
        const filename = `recording_${Date.now()}.wav`;
        const file = new File([this.recordedAudioBlob], filename, {
          type: "audio/wav",
        });
        formData.append("audio", file);
      } else {
        formData.append("audio", this.recordedAudioBlob);
      }

      // Update progress
      setTimeout(
        () => ParkinsonsApp.animateProgress(progressBar, 60, 1000),
        1000
      );

      // Send to server
      const response = await fetch("/upload_audio", {
        method: "POST",
        body: formData,
      });

      // Update progress
      setTimeout(
        () => ParkinsonsApp.animateProgress(progressBar, 90, 500),
        2000
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      // Complete progress
      ParkinsonsApp.animateProgress(progressBar, 100, 500);

      // Hide loading modal after a short delay
      setTimeout(() => {
        loadingModal.hide();
        this.displayResults(result);
      }, 1000);
    } catch (error) {
      console.error("Analysis error:", error);
      loadingModal.hide();

      let message = "Analiz sırasında hata oluştu.";
      if (error.message.includes("Failed to fetch")) {
        message = "Sunucuya bağlanılamadı. İnternet bağlantınızı kontrol edin.";
      } else if (error.message) {
        message = error.message;
      }

      ParkinsonsApp.showToast(message, "danger");
    } finally {
      removeLoading();
    }
  }

  displayResults(result) {
    // Show results section
    const resultsSection = document.getElementById("resultsSection");
    resultsSection.style.display = "block";

    // Scroll to results
    ParkinsonsApp.scrollToElement(resultsSection);

    // Update prediction results
    this.updatePredictionResults(result.prediction);

    // Update feature analysis
    this.updateFeatureAnalysis(
      result.features,
      result.healthy_averages,
      result.parkinson_averages
    );

    // Update charts
    this.updateCharts(result.feature_comparison_plot, result.radar_plot);

    // Add animations
    resultsSection.classList.add("fade-in-up");

    ParkinsonsApp.showToast(
      "Analiz tamamlandı! Sonuçları aşağıda görebilirsiniz.",
      "success"
    );
  }

  updatePredictionResults(prediction) {
    const predictionIcon = document.getElementById("predictionIcon");
    const predictionText = document.getElementById("predictionText");
    const predictionDescription = document.getElementById(
      "predictionDescription"
    );
    const probabilityBar = document.getElementById("probabilityBar");
    const probabilityText = document.getElementById("probabilityText");
    const riskLevel = document.getElementById("riskLevel");
    const confidenceBar = document.getElementById("confidenceBar");
    const confidenceText = document.getElementById("confidenceText");

    const isParkinson = prediction.prediction === 1;
    const probability = (prediction.probability * 100).toFixed(1);
    const confidence = (prediction.confidence * 100).toFixed(1);

    // Update icon and text
    if (isParkinson) {
      predictionIcon.innerHTML =
        '<i class="fas fa-exclamation-triangle fa-4x text-warning"></i>';
      predictionText.textContent = "Risk Tespit Edildi";
      predictionText.className = "prediction-text mb-2 text-warning";
      predictionDescription.textContent =
        "Ses analizinde Parkinson hastalığı riski tespit edilmiştir.";
    } else {
      predictionIcon.innerHTML =
        '<i class="fas fa-check-circle fa-4x text-success"></i>';
      predictionText.textContent = "Normal Bulgular";
      predictionText.className = "prediction-text mb-2 text-success";
      predictionDescription.textContent =
        "Ses analizinde normal bulgular tespit edilmiştir.";
    }

    // Update probability bar
    probabilityBar.className = `progress-bar ${
      isParkinson ? "bg-warning" : "bg-success"
    }`;
    ParkinsonsApp.animateProgress(probabilityBar, probability, 1500);
    probabilityText.textContent = `Olasılık: %${probability}`;

    // Update risk level
    const riskLevelClass =
      prediction.risk_level === "High"
        ? "danger"
        : prediction.risk_level === "Medium"
        ? "warning"
        : "success";
    const riskLevelText =
      prediction.risk_level === "High"
        ? "Yüksek Risk"
        : prediction.risk_level === "Medium"
        ? "Orta Risk"
        : "Düşük Risk";

    riskLevel.innerHTML = `
            <div class="alert alert-${riskLevelClass}">
                <i class="fas fa-info-circle me-2"></i>
                <strong>Risk Seviyesi:</strong> ${riskLevelText}
                <br>
                <small>Bu analiz tıbbi tanı değildir. Mutlaka bir doktora danışınız.</small>
            </div>
        `;

    // Update confidence bar
    ParkinsonsApp.animateProgress(confidenceBar, confidence, 1500);
    confidenceText.textContent = `Güven: %${confidence}`;
  }

  updateFeatureAnalysis(features, healthyAverages, parkinsonAverages) {
    const frequencyFeatures = document.getElementById("frequencyFeatures");
    const qualityFeatures = document.getElementById("qualityFeatures");
    const complexityFeatures = document.getElementById("complexityFeatures");

    // Feature categories
    const categories = {
      frequency: [
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)",
        "MDVP:RAP",
      ],
      quality: ["MDVP:Shimmer", "NHR", "HNR", "MDVP:APQ", "Shimmer:DDA"],
      complexity: ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
    };

    // Create feature items
    function createFeatureItems(featureNames, container) {
      container.innerHTML = "";
      featureNames.forEach((featureName) => {
        if (features[featureName] !== undefined) {
          const featureItem = document.createElement("div");
          featureItem.className = "feature-item";

          const value = features[featureName];
          const formattedValue = ParkinsonsApp.formatFeatureValue(
            value,
            featureName
          );

          featureItem.innerHTML = `
                        <span class="feature-name">${featureName}</span>
                        <span class="feature-value">${formattedValue}</span>
                    `;

          container.appendChild(featureItem);
        }
      });
    }

    createFeatureItems(categories.frequency, frequencyFeatures);
    createFeatureItems(categories.quality, qualityFeatures);
    createFeatureItems(categories.complexity, complexityFeatures);
  }

  updateCharts(featureComparisonPlot, radarPlot) {
    // Plot feature comparison chart
    if (featureComparisonPlot) {
      const plotData = JSON.parse(featureComparisonPlot);
      Plotly.newPlot("featureComparisonPlot", plotData.data, plotData.layout, {
        responsive: true,
        displayModeBar: false,
      });
    }

    // Plot radar chart
    if (radarPlot) {
      const radarData = JSON.parse(radarPlot);
      Plotly.newPlot("radarPlot", radarData.data, radarData.layout, {
        responsive: true,
        displayModeBar: false,
      });
    }
  }
}

// Initialize audio recorder when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  window.audioRecorder = new AudioRecorder();
});
