document.addEventListener("DOMContentLoaded", function () {
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

            if (container && preview) {
                container.style.display = 'block';
                preview.src = reader.result;
                document.getElementById('fullSizePreview').src = reader.result;
            }
        };

        if (this.files.length > 0) {
            reader.readAsDataURL(this.files[0]);
        }
    });

    // Handle Form Submission and Redirect
    document.getElementById('uploadForm').addEventListener('submit', async function (e) {
        e.preventDefault();

        const submitButton = document.querySelector('button[type="submit"]');
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        submitButton.disabled = true;

        // Simulated API call (Replace this with actual API fetch)
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

        // Redirect to result page
        window.location.href = "result.html"; // Ensure result.html exists

        // Reset button state (not needed if redirect happens)
        submitButton.innerHTML = 'Analyze Image';
        submitButton.disabled = false;
    });
});
