document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(error => {
                throw new Error(error.error || 'Unknown error');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        // Handle success response
    })
    .catch(error => {
        console.error('Error:', error.message);
        // Handle error response
    });
});
