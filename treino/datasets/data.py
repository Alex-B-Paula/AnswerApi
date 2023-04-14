# Get the train and validation json file in the HF script format
# inspiration: file squad.py at https://github.com/huggingface/datasets/tree/master/datasets/squad
import json
import numpy as np

files = ['squad-train-v1.1.json', 'squad-dev-v1.1.json']

for file in files:

    # Opening JSON file & returns JSON object as a dictionary
    f = open(file, encoding="utf-8")
    data = json.load(f)

    # Iterating through the json list
    entry_list = list()
    id_list = list()

    for row in data['data']:
        title = row['title']

        for paragraph in row['paragraphs']:
            context = paragraph['context']

            for qa in paragraph['qas']:
                entry = {}

                qa_id = qa['id']
                question = qa['question']
                answers = qa['answers']

                entry['id'] = qa_id
                # entry['title'] = title.strip()
                # entry['context'] = context.strip()
                # entry['question'] = question.strip()

                entry['input_ids'] = 'question: %s  context: %s' % (question.strip(), context.strip())

                answer_starts = [answer["answer_start"] for answer in answers]

                # keep unique texts
                answer_texts = [answer["text"].strip() for answer in answers]
                sorted_values, index_values = np.unique(answer_texts, return_index=True)
                answer_texts = (np.array(answer_texts)[index_values]).tolist()
                answer_starts = (np.array(answer_starts)[index_values]).tolist()

                # if len(answer_starts) > 1:
                #   print(qa_id)

                entry['answers'] = {}
                entry['answers']['answer_start'] = answer_starts
                entry['answers']['text'] = answer_texts

                # entry['labels'] = '%s' % answer_texts

                entry_list.append(entry)

    reverse_entry_list = entry_list[::-1]

    # for entries with same id, keep only last one (corrected texts by he group Deep Learning Brasil)
    unique_ids_list = list()
    unique_entry_list = list()
    for entry in reverse_entry_list:
        qa_id = entry['id']
        if qa_id not in unique_ids_list:
            unique_ids_list.append(qa_id)
            unique_entry_list.append(entry)

    # Closing file
    f.close()

    new_dict = {}
    new_dict['data'] = unique_entry_list

    file_name = 'pt_' + str(file)
    with open(file_name, 'w') as json_file:
        json.dump(new_dict, json_file)