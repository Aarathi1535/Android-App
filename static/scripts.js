document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const formData = new FormData();
        const files = document.getElementById('fileInput').files;
        const prompt = document.getElementById('promptInput').value;

        if (files.length === 0) {
            resultDiv.innerHTML = '<p>No files selected.</p>';
            return;
        }

        if (!prompt.trim()) {
            resultDiv.innerHTML = '<p>No prompt provided.</p>';
            return;
        }

        // Append files and prompt to FormData
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        formData.append('prompt', prompt);

        try {
            // Send the form data using fetch
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json(); // Expect JSON response
                if (data.status === 'success') {
                    // Redirect to results page
                    window.location.href = '/result';
                } else {
                    resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                }
            } else {
                // Handle non-JSON responses (e.g., HTML error pages)
                const errorText = await response.text();
                resultDiv.innerHTML = `<p>Error: ${errorText}</p>`;
            }
        } catch (error) {
            resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
        }
    });
});
