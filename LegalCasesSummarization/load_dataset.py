import os


original_training = 'docs/extracted'
summaries_training = 'docs/summaries'
original_test = 'docs/testing/extracted'
summaries_test = 'docs/testing/summaries'
training_dataset = []
test_dataset = []


def read_training_dataset(start = 0, end = 25):
    training_dataset = []
    print('all files -',os.listdir(original_training))
    for original_filename in os.listdir(original_training):
        original_file = os.path.join(original_training, original_filename)
        print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        # print(text)
        json_object = {"original": text, "summary": ""}
        training_dataset.append(json_object)

    i = 0
    for summary_filename in os.listdir(summaries_training):
        if summary_filename not in os.listdir(original_training):
            continue
        original_file = os.path.join(summaries_training, summary_filename)
        print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        training_dataset[i]['summary'] = text
        i += 1

    # print(training_dataset)
    return training_dataset[start:end]


def read_test_dataset():
    training_dataset = []
    for original_filename in os.listdir(original_test):
        original_file = os.path.join(summaries_test, original_filename)
        print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        # print(text)
        json_object = {"original": text, "summary": ""}
        training_dataset.append(json_object)

    i = 0
    for summary_filename in os.listdir(summaries_test):
        if summary_filename not in os.listdir(original_test):
            continue
        original_file = os.path.join(summaries_test, summary_filename)
        print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        training_dataset[i]['summary'] = text
        i += 1

    # print(training_dataset)
    return training_dataset






def read_training_dataset_optimized(start = 0, end = 25):
    # print('all files -',os.listdir(original_training))
    training_dataset = []
    for original_filename in os.listdir(original_training)[start:end]:
        original_file = os.path.join(original_training, original_filename)
        # print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        # print(text)
        json_object = {"original": text, "summary": ""}
        training_dataset.append(json_object)

    i = 0
    for summary_filename in os.listdir(summaries_training)[start:end]:
        if summary_filename not in os.listdir(original_training):
            continue
        original_file = os.path.join(summaries_training, summary_filename)
        # print(original_file)
        of = open(original_file, 'r', encoding='utf-8')
        text = of.read()
        text = text.replace("\n", "")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        training_dataset[i]['summary'] = text
        i += 1

    # print(training_dataset)
    return training_dataset

