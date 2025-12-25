def seqtolist(text_area_content, file):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # 定义处理数据的函数
    def parse_data(content):
        lines = content.strip().split('>')
        for line in lines:
            if line:
                parts = line.split('|')
                label, data_type = parts[0].strip(), parts[1].split('\n')[0].strip()
                sequence = parts[1].split('\n', 1)[1].replace('\n', '').replace('\r', '')

                if data_type == 'train':
                    train_data.append(sequence)
                    train_label.append(int(label))
                elif data_type == 'test':
                    test_data.append(sequence)
                    test_label.append(int(label))

    # 优先使用文件中的数据
    if file:
        content = file.read().decode('utf-8')
        parse_data(content)
    elif text_area_content:
        parse_data(text_area_content)

    return train_data, train_label, test_data, test_label
