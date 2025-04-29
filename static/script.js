document.addEventListener("DOMContentLoaded", function () {
    // ----- Code for the Index Page  -----
    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        const uploadBox = document.querySelector(".upload-area");
        if (uploadBox) {
            uploadBox.addEventListener("click", function () {
                document.getElementById("imageInput").click();
            });
        }

        // Handle Image Upload Preview
        document.getElementById('imageInput').addEventListener('change', function (e) {
            const reader = new FileReader();
            reader.onload = function () {
                const container = document.getElementById('thumbnailContainer');
                const preview = document.getElementById('imagePreview');
                const placeholder = document.getElementById('previewPlaceholder');

                if (container && preview) {
                    preview.src = reader.result;
                    preview.style.display = 'block'; // Show the preview image
                    if (placeholder) {
                        placeholder.style.display = 'none'; // Hide the placeholder text
                    }
                    document.getElementById('fullSizePreview').src = reader.result;
                    // Store the uploaded image in sessionStorage for the result page
                    sessionStorage.setItem("uploadedImage", reader.result);
                }
            };

            if (this.files.length > 0) {
                reader.readAsDataURL(this.files[0]);
            }
        });

        // ----- Handle Form Submission and Redirect with Processing Overlay -----  
        uploadForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const submitButton = document.querySelector('button[type="submit"]');
            const percentageText = document.querySelector('.percentage-text');
            const loadingOverlay = document.getElementById("loadingOverlay");

            // Show loading overlay on click
            loadingOverlay.style.display = "flex";
            submitButton.innerHTML = '<span class="spinner-grow spinner-grow-sm me-2" role="status" aria-hidden="true"></span>Analyzing...';
            submitButton.classList.add('disabled');
            submitButton.disabled = true;

            // Create EventSource for SSE
            const eventSource = new EventSource('/progress');

            // Handle progress updates
            eventSource.onmessage = function (event) {
                const data = JSON.parse(event.data);
                const progress = Math.round(data.progress * 100);
                percentageText.textContent = `${progress}%`;

                // If progress is 100%, close the connection and prepare for redirect
                if (progress >= 100) {
                    eventSource.close();
                }
            };

            // Handle errors
            eventSource.onerror = function () {
                eventSource.close();
                alert('Error receiving progress updates');
                loadingOverlay.style.display = "none";
                submitButton.innerHTML = 'Click to Analyze';
                submitButton.disabled = false;
            };

            // ----- Handle Form Submission -----   
            try {
                const formData = new FormData();
                const fileInput = document.getElementById('imageInput');

                if (fileInput.files.length === 0) {
                    alert('Please select an image to analyze');
                    eventSource.close();
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                    return;
                }

                formData.append('file', fileInput.files[0]);

                // Submit to Flask API
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    alert(result.error);
                    eventSource.close();
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                    return;
                }

                // ----- Handle Redirect -----  
                if (result.success && result.redirect) {
                    // Wait for progress to reach 100% through SSE
                    await new Promise(resolve => {
                        const checkInterval = setInterval(() => {
                            if (parseInt(percentageText.textContent) >= 100) {
                                clearInterval(checkInterval);
                                resolve();
                            }
                        }, 10);
                    });

                    // Show 100% for a moment before redirecting
                    await new Promise(resolve => setTimeout(resolve, 100));

                    // Redirect to the result page
                    window.location.href = result.redirect;
                } else {
                    // Handle unexpected response
                    alert('An error occurred during image analysis');
                    eventSource.close();
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during image analysis');
                eventSource.close();
                loadingOverlay.style.display = "none";
                submitButton.innerHTML = 'Click to Analyze';
                submitButton.disabled = false;
            }
        });
    }

    // ----- Code for the Result Page -----
    const resultContainer = document.getElementById("resultContainer");
    if (resultContainer) {
        const resultText = document.getElementById("resultText");
        const userImage = document.getElementById("userImage");
        const introText = document.getElementById("introText");

        // Get the filename from the URL
        const pathSegments = window.location.pathname.split('/');
        const filename = pathSegments[pathSegments.length - 1];

        if (userImage) {
            userImage.src = `/static/uploads/${filename}`;
        }

        // Fetch the detection results
        fetch(`/api/analyze`, {
            method: 'POST',
            body: (() => {
                const formData = new FormData();
                formData.append('file', filename);
                return formData;
            })()
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultText.innerHTML = `<span class="error">Error: ${data.error}</span>`;
                    return;
                }

                // ----- Set the result text based on the ai model prediction -----
                const isAi = data.classification === 'AI-generated Image';
                const colorClass = isAi ? "fraudulent" : "authentic";
                const confidence = isAi ? data.ai_probability.toFixed(2) : data.real_probability.toFixed(2);

                resultText.innerHTML = `<span class="${colorClass}">${confidence}% confidence - ${data.classification}</span>`;
            })

            // ----- Handle Error -----
            .catch(error => {
                console.error('Error:', error);
                resultText.innerHTML = '<span class="error">Error retrieving analysis results</span>';
            });

    }
    // Add Find More button functionality
    const findMoreBtn = document.getElementById("findMoreBtn");
    if (findMoreBtn) {
        findMoreBtn.addEventListener("click", function () {
            // Disable button and show loading state
            findMoreBtn.disabled = true;
            findMoreBtn.textContent = "Generating...";
            findMoreBtn.style.display = "none";

            // Get the current image filename from URL or path
            const pathSegments = window.location.pathname.split('/');
            const filename = pathSegments[pathSegments.length - 1];

            // Call the API to generate Grad-CAM with default alpha value
            generateGradCam(filename, 0.5);
        });
    }

    // Function to generate Grad-CAM with specified alpha value
    function generateGradCam(filename, alpha) {
        fetch('/api/grad-cam', {
            method: 'POST',
            body: (() => {
                const formData = new FormData();
                formData.append('file', filename);
                formData.append('alpha', alpha);
                return formData;
            })()
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(`Error: ${data.error}`);
                    findMoreBtn.textContent = "Find More";
                    findMoreBtn.disabled = false;
                    return;
                }

                // Create or update the Grad-CAM container
                let gradcamContainer = document.getElementById("gradcamContainer");
                if (!gradcamContainer) {
                    gradcamContainer = document.createElement("div");
                    gradcamContainer.id = "gradcamContainer";
                    gradcamContainer.className = "mt-4 text-center";

                    // Insert after the result text
                    const resultText = document.getElementById("resultText");
                    resultText.parentNode.insertBefore(gradcamContainer, resultText.nextSibling);
                }

                // Display the Grad-CAM visualization
                gradcamContainer.innerHTML = `
                <h4 class="mt-3">Explainability Visualization</h4>
                <p>Areas highlighted in red/yellow influenced the model's decision the most</p>
                <img src="${data.gradcam_url}?t=${new Date().getTime()}" class="img-fluid mt-2" id="gradcamImage" alt="Grad-CAM Visualization">
            `;

                // Show the slider container
                const sliderContainer = document.getElementById("sliderContainer");
                if (sliderContainer) {
                    sliderContainer.style.display = "block";
                }

                // Reset button
                findMoreBtn.textContent = "Find More";
                findMoreBtn.disabled = false;

                // Set up slider event listener
                setupSlider(filename);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error generating visualization');
                findMoreBtn.textContent = "Find More";
                findMoreBtn.disabled = false;
            });
    }

    // Function to set up slider event handling
    function setupSlider(filename) {
        const alphaSlider = document.getElementById("alphaSlider");
        if (alphaSlider) {
            // Reset slider to middle position
            alphaSlider.value = 0.5;
            
            // Debounce to prevent too many requests
            let debounceTimer;
            
            // Add visual feedback class
            const gradcamContainer = document.getElementById("gradcamContainer");
            if (gradcamContainer) {
                gradcamContainer.classList.add("has-slider");
            }
            
            alphaSlider.addEventListener("input", function () {
                // Visual feedback during slider movement
                const gradcamImage = document.getElementById("gradcamImage");
                if (gradcamImage) {
                    gradcamImage.style.opacity = 0.7;
                    gradcamImage.classList.add("updating");
                }
                
                // Show a loading indicator
                let loadingIndicator = document.getElementById("sliderLoadingIndicator");
                if (!loadingIndicator) {
                    loadingIndicator = document.createElement("div");
                    loadingIndicator.id = "sliderLoadingIndicator";
                    loadingIndicator.className = "text-center mt-2";
                    loadingIndicator.innerHTML = '<small>Updating visualization...</small>';
                    const sliderContainer = document.getElementById("sliderContainer");
                    if (sliderContainer) {
                        sliderContainer.appendChild(loadingIndicator);
                    }
                }

                // Clear any previous timer
                clearTimeout(debounceTimer);

                // Set a new timer
                debounceTimer = setTimeout(() => {
                    const alpha = parseFloat(alphaSlider.value);
                    console.log("Generating GradCAM with alpha:", alpha);

                    // Get currently displayed image filename if not provided
                    const currentFilename = filename || (() => {
                        const pathSegments = window.location.pathname.split('/');
                        return pathSegments[pathSegments.length - 1];
                    })();

                    // Generate new Grad-CAM with current alpha
                    fetch('/api/grad-cam', {
                        method: 'POST',
                        body: (() => {
                            const formData = new FormData();
                            formData.append('file', currentFilename);
                            formData.append('alpha', alpha);
                            return formData;
                        })()
                    })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! Status: ${response.status}`);
                            }
                            return response.json();
                        })
                        .then(data => {
                            if (data.error) {
                                console.error('Error:', data.error);
                                throw new Error(data.error);
                            }

                            // Update the image with a cache-busting parameter
                            const gradcamImage = document.getElementById("gradcamImage");
                            if (gradcamImage) {
                                const cacheBuster = new Date().getTime();
                                gradcamImage.src = `${data.gradcam_url}&cb=${cacheBuster}`;
                                
                                // Remove updating class when the image loads
                                gradcamImage.onload = function() {
                                    gradcamImage.style.opacity = 1.0;
                                    gradcamImage.classList.remove("updating");
                                    
                                    // Remove loading indicator
                                    const loadingIndicator = document.getElementById("sliderLoadingIndicator");
                                    if (loadingIndicator) {
                                        loadingIndicator.remove();
                                    }
                                };
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            
                            // Reset image opacity even on error
                            const gradcamImage = document.getElementById("gradcamImage");
                            if (gradcamImage) {
                                gradcamImage.style.opacity = 1.0;
                                gradcamImage.classList.remove("updating");
                            }
                            
                            // Remove loading indicator and show error
                            const loadingIndicator = document.getElementById("sliderLoadingIndicator");
                            if (loadingIndicator) {
                                loadingIndicator.innerHTML = `<small class="text-danger">Error: ${error.message}</small>`;
                                
                                // Auto-hide error after a few seconds
                                setTimeout(() => {
                                    if (loadingIndicator.parentNode) {
                                        loadingIndicator.remove();
                                    }
                                }, 3000);
                            }
                        });
                }, 300); // 300ms debounce
            });
        }
    }


    // Function to open the full-size image modal
    window.openFullSizeImage = function () {
        const modal = new bootstrap.Modal(document.getElementById('imageModal'));
        modal.show();
    };
});