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
    
        // Handle Form Submission and Redirect with Processing Overlay
        uploadForm.addEventListener('submit', async function (e) {
            e.preventDefault();
    
            const submitButton = document.querySelector('button[type="submit"]');
            // Show loading overlay on click
            document.getElementById("loadingOverlay").style.display = "flex";
    
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            submitButton.disabled = true;
    
            try {
                // Create FormData with the file from input
                const formData = new FormData();
                const fileInput = document.getElementById('imageInput');
                
                if (fileInput.files.length === 0) {
                    alert('Please select an image to analyze');
                    document.getElementById("loadingOverlay").style.display = "none";
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
                    alert('An error occurred during image analysis');
                    document.getElementById("loadingOverlay").style.display = "none";
                    submitButton.innerHTML = 'Click to Analyze';
                    submitButton.disabled = false;
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during image analysis');
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
            // introText.textContent = "Based on our analysis, this image is...";
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
});