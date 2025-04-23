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
    
            // Start percentage animation
            let percentage = 0;
            const interval = setInterval(() => {
                percentage += 1;
                if (percentage <= 100) {
                    percentageText.textContent = `${percentage}%`;
                }
            }, 50); // Adjust speed by changing the interval

            // ----- Handle Form Submission -----   
            try {
                const formData = new FormData();
                const fileInput = document.getElementById('imageInput');
                
                if (fileInput.files.length === 0) {
                    alert('Please select an image to analyze');
                    clearInterval(interval);
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                    return;
                }
                
                formData.append('file', fileInput.files[0]);
                
                // Submit to Flask API
                const response = await fetch('/uploads', {
                    method: 'POST', 
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    alert(result.error);
                    clearInterval(interval);
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                    return;
                }
                
                // ----- Handle Redirect -----  
                if (result.success && result.redirect) {
                    // Wait for percentage to reach 100%
                    if (percentage < 100) {
                        await new Promise(resolve => {
                            const checkInterval = setInterval(() => {
                                if (percentage >= 100) {
                                    clearInterval(checkInterval);
                                    resolve();
                                }
                            }, 10);
                        });
                    }
                    
                    // Show 100% for a moment before redirecting
                    percentageText.textContent = "100%";
                    await new Promise(resolve => setTimeout(resolve, 100)); 
                    
                    // Redirect to the result page
                    window.location.href = result.redirect;
                } else {
                    // Handle unexpected response
                    alert('An error occurred during image analysis');
                    clearInterval(interval);
                    loadingOverlay.style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during image analysis');
                clearInterval(interval);
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
            
            resultText.innerHTML = `<span class="${colorClass}">${confidence}% confidence - ${data.classification}</span>`; })

        // ----- Handle Error -----
        .catch(error => {
            console.error('Error:', error);
            resultText.innerHTML = '<span class="error">Error retrieving analysis results</span>';
        });

    }
     // Add Find More button functionality
     const findMoreBtn = document.getElementById("findMoreBtn");
     if (findMoreBtn) {
        findMoreBtn.addEventListener("click", function() {
            // Disable the button immediately to prevent double-clicks
            this.disabled = true;
            // Optionally change text to show processing
            this.textContent = "Generating...";
            
            // Get the current image filename from URL or another source
            const pathSegments = window.location.pathname.split('/');
            const filename = pathSegments[pathSegments.length - 1];
            
            // Call the API to generate Grad-CAM
            fetch('/api/grad-cam', {
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
                    alert(`Error: ${data.error}`);
                    // Re-enable button on error
                    findMoreBtn.disabled = false;
                    findMoreBtn.textContent = "Find More";
                    return;
                }
                
                // Create or update the Grad-CAM container
                let gradcamContainer = document.getElementById("gradcamContainer");
                if (!gradcamContainer) {
                    gradcamContainer = document.createElement("div");
                    gradcamContainer.id = "gradcamContainer";
                    gradcamContainer.className = "text-center mt-3";
                    
                    // Add container after result text
                    const resultText = document.getElementById("resultText");
                    resultText.parentNode.insertBefore(gradcamContainer, resultText.nextSibling);
                }
                
                // Display the Grad-CAM visualization
                gradcamContainer.innerHTML = `
                    <h4>Prediction Explanation</h4>
                    <img src="${data.gradcam_url}" class="img-fluid" alt="Grad-CAM Visualization">
                `;
                
                // Important: EITHER hide the button completely
                findMoreBtn.style.display = "none";
                
                // OR keep it disabled permanently
                // findMoreBtn.disabled = true;
                // findMoreBtn.textContent = "Analysis Complete";
            })
            .catch(error => {
                console.error('Error:', error);
                // Re-enable button on error
                findMoreBtn.disabled = false;
                findMoreBtn.textContent = "Find More";
            });
        });
    }   
        
    // Function to open the full-size image modal
    window.openFullSizeImage = function() {
        const modal = new bootstrap.Modal(document.getElementById('imageModal'));
        modal.show();
    };
});