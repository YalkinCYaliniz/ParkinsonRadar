// Main JavaScript for Parkinson Voice Analysis
document.addEventListener("DOMContentLoaded", function () {
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  // Add loading animation to buttons
  function addLoadingToButton(button, text = "Yükleniyor...") {
    const originalText = button.innerHTML;
    button.innerHTML = `<i class="spinner-border spinner-border-sm me-2" role="status"></i>${text}`;
    button.disabled = true;

    return function removeLoading() {
      button.innerHTML = originalText;
      button.disabled = false;
    };
  }

  // Show success toast
  function showToast(message, type = "success") {
    // Create toast element
    const toastHTML = `
            <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

    // Add to toast container (create if doesn't exist)
    let toastContainer = document.querySelector(".toast-container");
    if (!toastContainer) {
      toastContainer = document.createElement("div");
      toastContainer.className =
        "toast-container position-fixed top-0 end-0 p-3";
      toastContainer.style.zIndex = "1080";
      document.body.appendChild(toastContainer);
    }

    toastContainer.insertAdjacentHTML("beforeend", toastHTML);

    // Show toast
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, {
      autohide: true,
      delay: 5000,
    });
    toast.show();

    // Remove from DOM after hiding
    toastElement.addEventListener("hidden.bs.toast", function () {
      toastElement.remove();
    });
  }

  // Loading progress animation
  function animateProgress(progressBar, targetWidth, duration = 1000) {
    let currentWidth = 0;
    const increment = targetWidth / (duration / 16); // 60fps

    function updateProgress() {
      currentWidth += increment;
      if (currentWidth >= targetWidth) {
        currentWidth = targetWidth;
        progressBar.style.width = currentWidth + "%";
        progressBar.setAttribute("aria-valuenow", currentWidth);
        return;
      }

      progressBar.style.width = currentWidth + "%";
      progressBar.setAttribute("aria-valuenow", currentWidth);
      requestAnimationFrame(updateProgress);
    }

    requestAnimationFrame(updateProgress);
  }

  // Format time duration
  function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes
      .toString()
      .padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
  }

  // Format feature values
  function formatFeatureValue(value, featureName) {
    if (typeof value !== "number") return "N/A";

    // Different formatting for different feature types
    if (featureName.includes("Hz")) {
      return value.toFixed(2) + " Hz";
    } else if (featureName.includes("%")) {
      return (value * 100).toFixed(3) + "%";
    } else if (featureName.includes("dB")) {
      return value.toFixed(2) + " dB";
    } else {
      return value.toFixed(4);
    }
  }

  // Scroll to element with offset for fixed navbar
  function scrollToElement(element, offset = 80) {
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - offset;

    window.scrollTo({
      top: offsetPosition,
      behavior: "smooth",
    });
  }

  // Add intersection observer for animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("fade-in-up");
      }
    });
  }, observerOptions);

  // Observe elements for animation
  document
    .querySelectorAll(".info-card, .stat-card, .chart-card")
    .forEach((el) => {
      observer.observe(el);
    });

  // Copy text to clipboard
  function copyToClipboard(text) {
    navigator.clipboard
      .writeText(text)
      .then(function () {
        showToast("Panoya kopyalandı!", "success");
      })
      .catch(function () {
        showToast("Kopyalama başarısız!", "danger");
      });
  }

  // Download data as JSON
  function downloadAsJSON(data, filename) {
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(dataBlob);
    link.download = filename;
    link.click();

    URL.revokeObjectURL(link.href);
  }

  // Validate audio file
  function validateAudioFile(file) {
    const validTypes = [
      "audio/wav",
      "audio/mp3",
      "audio/mpeg",
      "audio/m4a",
      "audio/mp4",
    ];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!validTypes.some((type) => file.type.includes(type.split("/")[1]))) {
      throw new Error(
        "Desteklenmeyen dosya formatı. Lütfen WAV, MP3 veya M4A dosyası seçin."
      );
    }

    if (file.size > maxSize) {
      throw new Error("Dosya boyutu çok büyük. Maksimum 50MB desteklenir.");
    }

    return true;
  }

  // Handle file upload errors
  function handleUploadError(error) {
    console.error("Upload error:", error);
    let message = "Dosya yükleme hatası.";

    if (error.message) {
      message = error.message;
    } else if (error.status === 413) {
      message = "Dosya boyutu çok büyük.";
    } else if (error.status === 415) {
      message = "Desteklenmeyen dosya formatı.";
    } else if (error.status >= 500) {
      message = "Sunucu hatası. Lütfen daha sonra tekrar deneyin.";
    }

    showToast(message, "danger");
  }

  // Browser compatibility check
  function checkBrowserCompatibility() {
    const features = {
      mediaRecorder: typeof MediaRecorder !== "undefined",
      getUserMedia:
        navigator.mediaDevices && navigator.mediaDevices.getUserMedia,
      audioContext:
        typeof (window.AudioContext || window.webkitAudioContext) !==
        "undefined",
      fetch: typeof fetch !== "undefined",
      promises: typeof Promise !== "undefined",
    };

    const unsupported = Object.keys(features).filter((key) => !features[key]);

    if (unsupported.length > 0) {
      showToast(
        "Tarayıcınız bazı özellikleri desteklemiyor. Modern bir tarayıcı kullanmanız önerilir.",
        "warning"
      );
      return false;
    }

    return true;
  }

  // Initialize app
  function initializeApp() {
    // Check browser compatibility
    checkBrowserCompatibility();

    // Add keyboard shortcuts
    document.addEventListener("keydown", function (e) {
      // Esc key to close modals
      if (e.key === "Escape") {
        const modals = document.querySelectorAll(".modal.show");
        modals.forEach((modal) => {
          const bsModal = bootstrap.Modal.getInstance(modal);
          if (bsModal) bsModal.hide();
        });
      }
    });

    // Add tooltips to elements with title attribute
    const tooltipTriggerList = [].slice.call(
      document.querySelectorAll("[title]")
    );
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    console.log("Parkinson Voice Analysis App initialized");
  }

  // Make functions globally available
  window.ParkinsonsApp = {
    addLoadingToButton,
    showToast,
    animateProgress,
    formatTime,
    formatFeatureValue,
    scrollToElement,
    copyToClipboard,
    downloadAsJSON,
    validateAudioFile,
    handleUploadError,
  };

  // Initialize the app
  initializeApp();
});
