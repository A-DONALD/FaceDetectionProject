const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const filePath = document.getElementById('file-path');
const uploadButton = document.getElementById('upload-btn');
const deleteButton = document.getElementById('delete-btn');
const preview = document.getElementById('preview');

let selectedFile = null;

// Gestion du drag & drop
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

// Gestion de la sélection de fichier
fileInput.addEventListener('change', () => {
    selectedFile = fileInput.files[0];
    if (selectedFile) {
        handleFile(selectedFile);
    }
});

function handleFile(file) {
    const validExtensions = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (!validExtensions.includes(file.type)) {
        alert("Veuillez sélectionner un fichier image valide (JPEG, PNG, GIF, WEBP).");
        filePath.textContent = "";
        preview.innerHTML = "<span>Aucune image sélectionnée</span>";
        return;
    }

    filePath.textContent = file.name;

    // Afficher l'aperçu de l'image
    const reader = new FileReader();
    reader.onload = (event) => {
        preview.innerHTML = `<img src="${event.target.result}" alt="Aperçu de l'image">`;
    };
    reader.readAsDataURL(file);
}

// Envoi du fichier au serveur Flask
uploadButton.addEventListener('click', () => {
    if (!selectedFile) {
        alert('Veuillez sélectionner un fichier.');
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
            alert('Fichier envoyé avec succès : ' + data.message);
        })
        .catch(error => {
            console.error('Erreur lors de l\'envoi :', error);
        });
});

// Suppréssion du dossier upload
deleteButton.addEventListener('click', () => {
    if (confirm("Êtes-vous sûr de vouloir supprimer le répertoire et son contenu ?")) {
        fetch('/delete', {
            method: 'DELETE'
        })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
            })
            .catch(error => {
                console.error('Erreur lors de la suppression :', error);
                alert("Erreur lors de la suppression du répertoire.");
            });
    }
});