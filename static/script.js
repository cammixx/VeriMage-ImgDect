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
    
            // Simulated API call (Replace with your actual API fetch)
            const formData = new FormData(this);
            // const response = await fetch('/api/analyze', { method: 'POST', body: formData });
            // const result = await response.json();
    
            // Simulated result for now
            const result = {
                is_fraud: Math.random() > 0.5,
                confidence: (Math.random() * 100).toFixed(2)
            };
    
            // Store result data in sessionStorage (Temporary Data Storage)
            sessionStorage.setItem("analysisResult", JSON.stringify(result));
    
            // Delay redirection for 2 seconds to let the loading overlay be visible
            setTimeout(() => {
                window.location.href = "result.html";
            }, 2000);
    
            // Reset button state (not needed if redirect happens)
            submitButton.innerHTML = 'Analyze Image';
            submitButton.disabled = false;
        });
    }
    
    // ----- Code for the Result Page -----
    const resultContainer = document.getElementById("resultContainer");
    if (resultContainer) {
        const resultText = document.getElementById("resultText");
        const userImage = document.getElementById("userImage");
        const introText = document.getElementById("introText");
    
        // Retrieve data from sessionStorage
        const resultData = JSON.parse(sessionStorage.getItem("analysisResult"));
        const uploadedImage = sessionStorage.getItem("uploadedImage");
    
        if (uploadedImage && userImage) {
            userImage.src = uploadedImage;
        }
    
        if (resultData) {
            let verdictText = resultData.is_fraud ? "AI-generated" : "Authentic";
            // Choose CSS class based on result for emphasis
            let colorClass = resultData.is_fraud ? "fraudulent" : "authentic";
            resultText.innerHTML = `<span class="${colorClass}">${resultData.confidence}% confidence - ${verdictText}</span>`;
            // Keep the introductory text fixed without appending verdict info
            introText.textContent = "Compare with our data, we say this image is...";
        } else {
            resultText.textContent = "No result found.";
            introText.textContent = "";
        }
    }
});