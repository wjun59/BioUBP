// 根据选中的算法加载相应的 HTML 文件加载进来
document.querySelectorAll('input[name="algorithm"]').forEach((radio) => {
    radio.addEventListener('change', function () {
        const selectedAlgorithm = this.value;
        const paramsContainer = document.getElementById('algorithm-params');
        // 根据选中的算法加载相应的 HTML 文件
        let url = `/static/dna_alg/${selectedAlgorithm}.html`; // 假设表单HTML放在static文件夹下
        // 使用 AJAX 动态加载HTML表单
        fetch(url)
            .then(response => response.text())
            .then(html => {
                // 清空当前的参数区域
                paramsContainer.innerHTML = '';
                // 插入新的HTML表单
                paramsContainer.innerHTML = html;
            })
            .catch(err => {
                console.error('Error loading form:', err);
                paramsContainer.innerHTML = '<p>Error loading form. Please try again later.</p>';
            });
    });
});


//这个是特征提取算法里面的js
//以下是DNA-DAC特征提取算法
function dac_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function dac_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

//这个是特征提取算法里面的js
//以下是DNA-DACC特征提取算法
function dacc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dacc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function dacc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dacc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

//这个是特征提取算法里面的js
//以下是DNA-DCC特征提取算法
function dcc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dcc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function dcc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="dcc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

//这个是特征提取算法里面的js
//以下是DNA-TAC特征提取算法
function tac_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function tac_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是DNA-TCC特征提取算法
function tcc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tcc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function tcc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tcc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是DNA-TACC特征提取算法
function tacc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tacc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function tacc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="tacc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是DNA-NMBAC特征提取算法
function nmbac_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="nmbac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function nmbac_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="nmbac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

//这个是特征提取算法里面的js
//以下是DNA-gAC特征提取算法
function gac_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="gac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function gac_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="gac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是DNA-MAC特征提取算法
function mac2_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="mac2_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function mac2_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="mac2_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

function mac3_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="mac3_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function mac3_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="mac3_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


