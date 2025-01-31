const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const filePath = document.getElementById('file-path');
const uploadButton = document.getElementById('upload-btn');
const evalButton = document.getElementById('eval-btn');
const deleteButton = document.getElementById('delete-btn');
const preview = document.getElementById('preview');

let selectedFile = null;

// drag & drop manage
dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');

    selectedFile = event.dataTransfer.files[0];
    if (selectedFile) {
        handleFile(selectedFile);
    }
});

// Manage the file selection
fileInput.addEventListener('change', () => {
    selectedFile = fileInput.files[0];
    if (selectedFile) {
        handleFile(selectedFile);
    }
});

// handle the file after insertion
function handleFile(file) {
    const validExtensions = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (!validExtensions.includes(file.type)) {
        alert("Please, select a valid image format type (JPEG, PNG, GIF, WEBP).");
        filePath.textContent = "";
        preview.innerHTML = "<span>Aucune image sélectionnée</span>";
        return;
    }

    filePath.textContent = file.name;

    // File preview
    const reader = new FileReader();
    reader.onload = (event) => {
        preview.innerHTML = `<img src="${event.target.result}" alt="Image preview">`;
    };
    reader.readAsDataURL(file);
}

// Send the file to the Flask server
uploadButton.addEventListener('click', () => {
    if (!selectedFile) {
        alert('Please, select a file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            alert('File successfully sent : ' + data.message);
        })
        .catch(error => {
            console.error('Error during submittion :', error);
        });
});

// Start evaluation
evalButton.addEventListener('click', () => {
    fetch('/evaluate', {
        method: 'GET'
    })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
        })
        .catch(error => {
            console.error('Error during evaluation:', error);
            alert("the evaluation didn't perform as excepted");
        });
});

// upload folder delete
deleteButton.addEventListener('click', () => {
    if (confirm("Do you really want to delete the upload folder and its content ?")) {
        fetch('/delete', {
            method: 'DELETE'
        })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
            })
            .catch(error => {
                console.error('Error during deletion:', error);
                alert("We cannot delete the folder, open the console for more information.");
            });
    }
});