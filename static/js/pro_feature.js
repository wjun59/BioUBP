
// 根据选中的算法加载相应的 HTML 文件加载进来
document.querySelectorAll('input[name="algorithm"]').forEach((radio) => {
    radio.addEventListener('change', function () {
        const selectedAlgorithm = this.value;
        const paramsContainer = document.getElementById('algorithm-params');
        // 根据选中的算法加载相应的 HTML 文件
        let url = `/static/pro_alg/${selectedAlgorithm}.html`; // 假设表单HTML放在static文件夹下
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
//以下是Pro-AC特征提取算法
function ac_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="ac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function ac_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="ac_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是Pro-ACc特征提取算法
function acc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="acc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function acc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="acc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}

//这个是特征提取算法里面的js
//以下是Pro-cc特征提取算法
function cc_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="cc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function cc_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="cc_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是Pro-PCPseAACGeneral特征提取算法
function PCPseAACGeneral_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="PCPseAACGeneral_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function PCPseAACGeneral_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="PCPseAACGeneral_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是Pro-SCPseAACGeneral特征提取算法
function SCPseAACGeneral_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="SCPseAACGeneral_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function SCPseAACGeneral_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="SCPseAACGeneral_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是Pro-PCPseAAC特征提取算法
function PCPseAAC_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="PCPseAAC_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function PCPseAAC_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="PCPseAAC_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}


//这个是特征提取算法里面的js
//以下是Pro-SCPseAAC特征提取算法
function SCPseAAC_selectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="SCPseAAC_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
}

function SCPseAAC_deselectAllCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="SCPseAAC_param3"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}