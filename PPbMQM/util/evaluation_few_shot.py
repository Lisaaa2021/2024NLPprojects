import json
import ast
from nltk.tokenize import word_tokenize
import difflib
import jieba
import numpy as np

def extract_output_error_list_few_shot(json_data, target, source):
    """
    Take the error list from extract_json
    output a formatted error list for evaluation
    """

    #target = target.lower()
    target_list = word_tokenize(target)
    output_error_list = [] #list to store all error

    if json_data == [{}]:
        return output_error_list
    try:
        for i,j in enumerate(json_data):
            if type(j) == dict:
                error_type = j['error type'].lower()
                if error_type == 'accuracy':
                    severity_map = {1: 'minor',2:'minor',3:'minor',4:'major',5:'major'}
                    severity = severity_map[j['severity']]
                  #  if j['severity'] < 3:
                  #      continue

                else:
                    severity_map = {1: 'minor',2:'minor',3:'minor',4:'major',5:'major'}
                    severity = severity_map[j['severity']]
                  #  if j['severity'] < 3:
                  #      continue

                marked_text = j['marked text']
                explanation = j['explanation']
                system_index = j['error span index']
                marked_text_list = word_tokenize(marked_text)
                length = len(marked_text_list)

                if error_type == 'omission':
                    source_list = list(jieba.cut(source))
                    marked_text_list = list(jieba.cut(marked_text))
                    if marked_text in source:
                        for ind, word in enumerate(source_list):
                            if word == marked_text_list[0] and source_list[ind:ind+length] == marked_text_list:
                                # the step to get the index
                                start = ind
                                end = ind + length
                                output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}, explanation,j['severity']])

                elif marked_text in target and target.count(marked_text) == 1:
                    for ind, word in enumerate(target_list):
                        if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                            # the step to get the index
                            start = ind
                            end = ind + length - 1
                            output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}, explanation,j['severity']])

                elif marked_text in target and target.count(marked_text) > 1:
                    distance = 100000
                    for ind,word in enumerate(target_list):
                        if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                            # the step to get the index
                            if abs(system_index[0] - ind) < distance:
                                distance = abs(system_index[0] - ind)
                                start = ind
                                end = ind + length -1
                    # print('distance: ', distance)
                    # print(start)
                    # print()
                    output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}, explanation,j['severity']])


    except:
        output_error_list = None

    return output_error_list


import json
import ast
from nltk.tokenize import word_tokenize
import difflib
import jieba
import numpy as np

def extract_output_error_list_few_shot_production(json_data, target, source):
    """
    Take the error list from extract_json
    output a formatted error list for evaluation
    """

    #target = target.lower()
    target = str(target)
    source = str(source)
    target_list = word_tokenize(target)
    output_error_list = [] #list to store all error

    if json_data == [{}]:
        return output_error_list
    try:
        for i,j in enumerate(json_data):
            if type(j) == dict:
                error_type = j['error type'].lower()
                if error_type == 'accuracy':
                    severity_map = {1: 'minor',2:'minor',3:'minor',4:'minor',5:'major'}
                    severity = severity_map[j['severity']]
                    if j['severity'] < 3:
                        continue

                else:
                    severity_map = {1: 'minor',2:'minor',3:'minor',4:'major',5:'major'}
                    severity = severity_map[j['severity']]
                    if j['severity'] < 3:
                        continue

                marked_text = j['marked text']
           #     explanation = j['explanation']
                system_index = j['error span index']
                marked_text_list = word_tokenize(marked_text)
                length = len(marked_text_list)

                if error_type == 'omission':
                    source_list = list(jieba.cut(source))
                    marked_text_list = list(jieba.cut(marked_text))
                    if marked_text in source:
                        for ind, word in enumerate(source_list):
                            if word == marked_text_list[0] and source_list[ind:ind+length] == marked_text_list:
                                # the step to get the index
                                start = ind
                                end = ind + length
                                output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end},j['severity']])

                elif marked_text in target and target.count(marked_text) == 1:
                    for ind, word in enumerate(target_list):
                        if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                            # the step to get the index
                            start = ind
                            end = ind + length - 1
                            output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end},j['severity']])

                elif marked_text in target and target.count(marked_text) > 1:
                    distance = 100000
                    for ind,word in enumerate(target_list):
                        if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                            # the step to get the index
                            if abs(system_index[0] - ind) < distance:
                                distance = abs(system_index[0] - ind)
                                start = ind
                                end = ind + length -1
                    # print('distance: ', distance)
                    # print(start)
                    # print()
                    output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end},j['severity']])


    except:
        output_error_list = None

    return output_error_list
