<!--4个文件输入的js-->

// File size validation and display selected file name
document.getElementById('fileInput_traindata').addEventListener('change', function () {
    const maxSize = 3 * 1024 * 1024; // 3MB
    const file = this.files[0];

    if (file && file.size > maxSize) {
        alert(`File is too large. It should not exceed ${maxSize / (1024 * 1024)}MB.`);
        document.getElementById('fileInput_traindata').value = "";
        document.getElementById('traindatafileNameDisplay').textContent = '';
        return false;
    }
    // Check file format
    const allowedExtensions = ['csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        alert("Invalid file format. Only CSV files are allowed.");
        document.getElementById('fileInput_traindata').value = "";
        document.getElementById('traindatafileNameDisplay').textContent = '';
        return false;
    }

    var fileName = file ? file.name : '';
    document.getElementById('traindatafileNameDisplay').textContent = fileName ? fileName : '';
});


// File size validation and display selected file name
document.getElementById('fileInput_trainlables').addEventListener('change', function () {
    const maxSize = 3 * 1024 * 1024; // 3MB
    const file = this.files[0];

    if (file && file.size > maxSize) {
        alert(`File is too large. It should not exceed ${maxSize / (1024 * 1024)}MB.`);
        document.getElementById('fileInput_trainlables').value = "";
        document.getElementById('trainlablesfileNameDisplay').textContent = '';
        return false;
    }
    // Check file format
    const allowedExtensions = ['csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        alert("Invalid file format. Only CSV files are allowed.");
        document.getElementById('fileInput_traindata').value = "";
        document.getElementById('traindatafileNameDisplay').textContent = '';
        return false;
    }

    var fileName = file ? file.name : '';
    document.getElementById('trainlablesfileNameDisplay').textContent = fileName ? fileName : '';
});


// File size validation and display selected file name
document.getElementById('fileInput_testdata').addEventListener('change', function () {
    const maxSize = 3 * 1024 * 1024; // 3MB
    const file = this.files[0];

    if (file && file.size > maxSize) {
        alert(`File is too large. It should not exceed ${maxSize / (1024 * 1024)}MB.`);
        document.getElementById('fileInput_testdata').value = "";
        document.getElementById('testdatafileNameDisplay').textContent = '';
        return false;
    }
    // Check file format
    const allowedExtensions = ['csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        alert("Invalid file format. Only CSV files are allowed.");
        document.getElementById('fileInput_traindata').value = "";
        document.getElementById('traindatafileNameDisplay').textContent = '';
        return false;
    }

    var fileName = file ? file.name : '';
    document.getElementById('testdatafileNameDisplay').textContent = fileName ? fileName : '';
});


// File size validation and display selected file name
document.getElementById('fileInput_testlables').addEventListener('change', function () {
    const maxSize = 3 * 1024 * 1024; // 3MB
    const file = this.files[0];

    if (file && file.size > maxSize) {
        alert(`File is too large. It should not exceed ${maxSize / (1024 * 1024)}MB.`);
        document.getElementById('fileInput_testlables').value = "";
        document.getElementById('testlablesfileNameDisplay').textContent = '';
        return false;
    }
    // Check file format
    const allowedExtensions = ['csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        alert("Invalid file format. Only CSV files are allowed.");
        document.getElementById('fileInput_traindata').value = "";
        document.getElementById('traindatafileNameDisplay').textContent = '';
        return false;
    }

    var fileName = file ? file.name : '';
    document.getElementById('testlablesfileNameDisplay').textContent = fileName ? fileName : '';
});

<!--以上4个文件输入的js-->