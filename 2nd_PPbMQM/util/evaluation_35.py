from nltk.tokenize import word_tokenize
import json
import numpy as np
def extract_json_35(gpt_35_output):
    gpt_35_output = gpt_35_output.replace('```json', "").replace('```',"")
    try:
        json_data = json.loads(gpt_35_output)
        if type(json_data) == dict:
            if 'errors' in json_data.keys():
                return json_data['errors']
        else:
            return json_data
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return np.nan


def extract_output_error_list_35(output, target):
    json_data = extract_json_35(output)
    if type(json_data) == str:
        output_error_list = 'no json detected'
    else:
        target = target.lower()
        target_list = word_tokenize(target)
        output_error_list = [] #list to store all error
        for j in json_data:
           # error_span_index = j['error_span_index'] #暂时不用这个值
            severity = j['severity']
            error_type = j['error_type']
            marked_text = j['marked_text'].lower()
            explanation = j['explanation']

            marked_text_list = marked_text.split(' ')
            length = len(marked_text_list)

            if marked_text in target:
                marked_text_count = 0
                for ind, word in enumerate(target_list):
                    if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                        marked_text_count+=1
                        # the step to get the index
                        start = ind
                        end = ind + length
                        output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}, marked_text_count, explanation])

    return output_error_list
