document.addEventListener("DOMContentLoaded", function () {
    // ----- Code for the Index Page (Upload Form) -----
    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        const uploadBox = document.querySelector(".upload-area");
        if (uploadBox) {
            uploadBox.addEventListener("click", function () {
                document.getElementById("imageInput").click();
            });
        }
    
        // Handle Image Upload Preview and validation
        document.getElementById('imageInput').addEventListener('change', function (e) {
            const fileInput = this;
            const file = fileInput.files[0];
            
            // File validation
            if (file) {
                // Check file type
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!validTypes.includes(file.type)) {
                    showErrorMessage("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
                    fileInput.value = ''; // Clear the file input
                    return;
                }
                
                // Check file size (max 16MB)
                const maxSize = 16 * 1024 * 1024; // 16MB in bytes
                if (file.size > maxSize) {
                    showErrorMessage(`File is too large. Maximum size is 16MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB.`);
                    fileInput.value = ''; // Clear the file input
                    return;
                }
                
                // If validation passes, show preview
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
                
                reader.readAsDataURL(file);
            }
        });
    
        // Handle Form Submission and Redirect with Processing Overlay
        uploadForm.addEventListener('submit', async function (e) {
            e.preventDefault();
    
            const submitButton = document.querySelector('button[type="submit"]');
            const fileInput = document.getElementById('imageInput');
            
            // Check if a file is selected
            if (!fileInput.files || fileInput.files.length === 0) {
                showErrorMessage("Please select an image to analyze.");
                return;
            }
            
            // Show loading overlay on click
            document.getElementById("loadingOverlay").style.display = "flex";
    
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            submitButton.disabled = true;
    
            try {
                // Create FormData with the file from input
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                // Submit to our Flask API
                const response = await fetch('/upload', {
                    method: 'POST', 
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    showErrorMessage(result.error);
                    document.getElementById("loadingOverlay").style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                    return;
                }
                
                if (result.success && result.redirect) {
                    // Redirect to the result page
                    window.location.href = result.redirect;
                } else {
                    // Handle unexpected response
                    showErrorMessage('An error occurred during image analysis.');
                    document.getElementById("loadingOverlay").style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                }
            } catch (error) {
                console.error('Error:', error);
                showErrorMessage('An error occurred during image analysis. Please try again.');
                document.getElementById("loadingOverlay").style.display = "none";
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
            
            // Set the result text based on the prediction
            const isAi = data.classification === 'AI-generated Image';
            const colorClass = isAi ? "fraudulent" : "authentic";
            const confidence = isAi ? data.ai_probability.toFixed(2) : data.real_probability.toFixed(2);
            
            resultText.innerHTML = `<span class="${colorClass}">${confidence}% confidence - ${data.classification}</span>`;
            
            // Keep the introductory text
            introText.textContent = "Based on our analysis, this image is...";
        })
        .catch(error => {
            console.error('Error:', error);
            resultText.innerHTML = '<span class="error">Error retrieving analysis results</span>';
        });
    }
    
    // Function to open the full-size image modal
    window.openFullSizeImage = function() {
        const modal = new bootstrap.Modal(document.getElementById('imageModal'));
        modal.show();
    };
    
    // Function to show error messages
    function showErrorMessage(message) {
        // Check if we have a toast container, if not, create one
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        // Create unique ID for this toast
        const toastId = 'toast-' + Date.now();
        
        // Create toast HTML
        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header bg-danger text-white">
                    <strong class="me-auto">Error</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        // Add toast to container
        toastContainer.innerHTML += toastHtml;
        
        // Initialize and show the toast
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: 5000
        });
        toast.show();
        
        // Remove toast from DOM after it's hidden
        toastElement.addEventListener('hidden.bs.toast', function () {
            toastElement.remove();
        });
    }
});