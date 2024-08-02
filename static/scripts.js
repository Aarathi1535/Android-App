document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the form from submitting the traditional way

    var formData = new FormData();
    var files = document.getElementById('fileInput').files;
    var prompt = document.getElementById('promptInput').value;
    
    for (var i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    
    formData.append('prompt', prompt);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = 'Error: ' + data.error;
        } else {
            document.getElementById('result').innerText = 'Score: ' + data.score;
        }
    })
    .catch(error => {
        document.getElementById('result').innerText = 'Error: ' + error.message;
    });
});
