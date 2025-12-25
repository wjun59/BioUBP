const radioButtons = document.querySelectorAll('input[name="algorithm"]');
radioButtons.forEach((radio) => {
    radio.addEventListener('change', () => {
        document.querySelectorAll('.algorithm-btn').forEach((btn) => {
            btn.classList.remove('selected'); // 移除其他按钮的选中状态
        });
        radio.parentElement.classList.add('selected'); // 给选中的按钮添加类
    });
});

// 根据选中的算法加载相应的 HTML 文件加载进来
document.querySelectorAll('input[name="algorithm"]').forEach((radio) => {
    radio.addEventListener('change', function () {
        const selectedAlgorithm = this.value;
        const paramsContainer = document.getElementById('algorithm-params');
        // 根据选中的算法加载相应的 HTML 文件
        let url = `/load_algorithm/${selectedAlgorithm}`;

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


document.getElementById('submitButton_sampling').addEventListener('click', function (event) {
    const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked');
    if (!selectedAlgorithm) {
        alert('Please select an resampling algorithm.');
        event.preventDefault(); // Prevent form submission
    }
})


document.querySelectorAll('.model-button').forEach(label => {
    label.addEventListener('click', function () {
        // 移除所有按钮的选中样式
        document.querySelectorAll('.model-button').forEach(btn => btn.classList.remove('selected'));

        // 为当前标签添加选中样式
        this.classList.add('selected');
    });
});


function validateSelection() {
    // 获取所有 modelCheckbox 的输入元素
    const checkboxes = document.querySelectorAll('input[name="modelCheckbox"]');
    let isChecked = false;

    // 检查是否有选中的输入
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            isChecked = true;
        }
    });

    // 如果没有选中，给出警告并阻止提交
    if (!isChecked) {
        alert('Please select a model!');
        return false; // 阻止提交
    }
    return true; // 允许提交
}
// 在提交按钮或表单提交事件中调用此函数进行验证
document.getElementById('submitButton_sampling').addEventListener('click', function (event) {
    if (!validateSelection()) {

        event.preventDefault(); // 阻止按钮的默认行为
    }
});
