import json
import ast
from nltk.tokenize import word_tokenize
import difflib
import jieba
import numpy as np

#P A R S I N G #
def extract_json(output):
    """
    take gpt 4.0 system output
    return a list of mqm annotation
    """
    output = output.replace('```json', "").replace("```","") #.replace("'","")
    try:
        json_data = json.loads(output)
        if type(json_data) != dict and type(json_data) != list:
            print('something wrong !!')
        if type(json_data) == dict:
            result = []
            result.append(json_data)
            return result
        else:
            return json_data
    except json.JSONDecodeError as e:
        #print("Error decoding JSON:", e)
        result = np.nan
        return result



def extract_output_error_list(json_data, target, source):
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
                severity = j['severity'].lower()
                error_type = j['error type'].lower()
                marked_text = j['marked text']
#                explanation = j['explanation']
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
                                end = ind + length - 1
                                output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}])

                elif marked_text in target and target.count(marked_text) == 1:
                    for ind, word in enumerate(target_list):
                        if word == marked_text_list[0] and target_list[ind:ind+length] == marked_text_list:
                            # the step to get the index
                            start = ind
                            end = ind + length - 1
                            output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}])

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
                    output_error_list.append([error_type, severity,marked_text,{'start': start, 'end': end}])


    except:
        output_error_list = None

    return output_error_list




# def if_y_overlap_itself(y):
#     overlap = 'No'
#     if type(y) == list and len(y)>0:
#         all_spans = []
#         for sys_e in y:
#             if sys_e[3] == 0:
#                 continue
#             sys_span = list(range(sys_e[3]['start'],sys_e[3]['end']+1))
#             all_spans.extend(sys_span)
#         if len(all_spans) > len(set(all_spans)):
#             overlap ='Yes'
#     return overlap



def handle_no_error_segment(g_n, s_n):

    if g_n == 0 and s_n == 0:
        result = 'gold 0 system 0'
    elif g_n == 0 and s_n != 0:
        result = 'gold 0 system not'
    elif g_n != 0 and s_n == 0:
        result = 'gold not system 0'
    elif g_n != 0 and s_n != 0:
        result = 'gold not system not'

    return result

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a+size]


def get_recall(g , s, f):
    result = None
    if f == 'gold not system not':
        recall_list = []
        for gold_e in g:
            if gold_e[3] == 0:
                #recall_list.append(0)
                continue
            gold_text = gold_e[2]
            gold_span = set(range(gold_e[3]['start'],gold_e[3]['end']+1))

            longest_overlap_percentage = 0

            for sys_e in s:

                sys_text = sys_e[2]
                sys_span = set(range(sys_e[3]['start'],sys_e[3]['end']+1))

                token_overlap = gold_span.intersection(sys_span)

                if len(token_overlap) > 0: # only consider the case of overlap
                    overlap_s = get_overlap(sys_text, gold_text) # get the current overlap
                    c_overlap_percentage = len(overlap_s) / len(gold_text)
                    if c_overlap_percentage > longest_overlap_percentage:
                        longest_overlap_percentage = c_overlap_percentage


            if longest_overlap_percentage == 0:
                recall_list.append(0)
            else:
                recall_list.append(longest_overlap_percentage)

        if sum(recall_list) == 0:
            result = 'no overlap span'

        else:
            result = recall_list


    else:
        result = f
    return result


def get_precision(g , s, f):
    result = None
    if f == 'gold not system not':
        precision_list = []

        for sys_e in s:
            sys_text = sys_e[2]
            sys_start = sys_e[3]['start']
            sys_end = sys_e[3]['end']+1
            sys_span = set(range(sys_start, sys_end))


            longest_overlap_percentage = 0


            for ind,gold_e in enumerate(g):
                if gold_e[3] == 0:
                    continue
                gold_text = gold_e[2]
                gold_start = gold_e[3]['start']
                gold_end = gold_e[3]['end']+1
                gold_span = set(range(gold_start, gold_end))

                token_overlap = gold_span.intersection(sys_span)

                if len(token_overlap) > 0: # only consider the case of overlap
                    overlap_s = get_overlap(sys_text, gold_text) # get the current overlap
                    c_overlap_percentage = len(overlap_s) / len(sys_text)
                    if c_overlap_percentage > longest_overlap_percentage:
                        longest_ind = ind
                        longest_overlap_percentage = c_overlap_percentage

            if longest_overlap_percentage == 0:
                precision_list.append(0)
            else:
                precision_list.append(longest_overlap_percentage)


        if sum(precision_list) == 0:
            result = 'no overlap span'
        else:
            result = precision_list

    else:
        result = f

    return result




def get_segment_score(x):
    result = None
    if x == 'gold 0 system not':
        result = 0
    elif x == 'no overlap span':
        result = 0
    elif x == 'gold not system 0':
        result = 0
    elif x == 'gold 0 system 0':
        result = 1
    elif type(x) == list:
        result = round(sum(x)/len(x),3)
    return result







def get_severity_and_type_from_recall(g,s):
    error_type = []
    severity = []

    for gold_e in g:
        if gold_e[3] == 0:
            continue
        gold_type = gold_e[0]
        gold_severity = gold_e[1]
        gold_text = gold_e[2]
        gold_span = set(range(gold_e[3]['start'],gold_e[3]['end']+1))

        longest_overlap_percentage = 0
        longest_ind = None

        for ind,sys_e in enumerate(s):

            sys_text = sys_e[2]
            sys_span = set(range(sys_e[3]['start'],sys_e[3]['end']+1))

            token_overlap = gold_span.intersection(sys_span)

            if len(token_overlap) > 0: # only consider the case of overlap
                overlap_s = get_overlap(sys_text, gold_text) # get the current overlap
                c_overlap_percentage = len(overlap_s) / len(gold_text)
                if c_overlap_percentage > longest_overlap_percentage:
                    longest_ind = ind
                    longest_overlap_percentage = c_overlap_percentage


        if longest_ind != None:
            error_type.append([gold_type, s[longest_ind][0]])
            severity.append([gold_severity,s[longest_ind][1]])

    return error_type, severity


def get_severity_and_type_from_precision(g,s):
    error_type = []
    severity = []
    for sys_e in s:
        sys_type = sys_e[0]
        sys_severity = sys_e[1]
        sys_text = sys_e[2]
        sys_start = sys_e[3]['start']
        sys_end = sys_e[3]['end']+1
        sys_span = set(range(sys_start, sys_end))


        longest_overlap_percentage = 0
        longest_ind = None

        for ind,gold_e in enumerate(g):
            if gold_e[3] == 0:
                continue
            gold_text = gold_e[2]
            gold_start = gold_e[3]['start']
            gold_end = gold_e[3]['end']+1
            gold_span = set(range(gold_start, gold_end))

            token_overlap = gold_span.intersection(sys_span)

            if len(token_overlap) > 0: # only consider the case of overlap
                overlap_s = get_overlap(sys_text, gold_text) # get the current overlap
                c_overlap_percentage = len(overlap_s) / len(sys_text)
                if c_overlap_percentage > longest_overlap_percentage:
                    longest_ind = ind
                    longest_overlap_percentage = c_overlap_percentage

        if longest_ind != None:
            error_type.append([sys_type, g[longest_ind][0]])
            severity.append([sys_severity, g[longest_ind][1]])

    return error_type, severity


def extract_json_llama(output):
    """
    take system output
    return a list of mqm annotation
    """

    start = '''[\n    {'''
    start_index = output.index(start)
    end_index = output.index('''}\n]''') + 3
    new_output = output[start_index:end_index]

    try:
        return json.loads(new_output)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        result = np.nan
        return result


# def match_index(s):
#     try:
#         if type(s) == list and len(s) > 0:
#             repeated_marked_text = set()

#             for ind, ss in enumerate(s):
#                 if ss[4] > 1:
#                     repeated_marked_text.add(ss[2])

#             if len(repeated_marked_text) < 1:
#                 return s
#             else:
#                 print(s)
#                 remove_ind = []
#                 remove_value = []
#                 for r in repeated_marked_text:
#                     distance = 100000
#                     for ind, ss in enumerate(s):
#                         if ss[2] == r: # for this marked_text:
#                             remove_ind.append(ind)
#                             r_start = ss[3]['start']
#                             s_start = ss[6][0]
#                             if abs(r_start - s_start) < distance:
#                                 distance = abs(r_start - s_start)
#                                 ind_keep = ind # update index with the shortest distance
#                     remove_ind.remove(ind_keep)

#                 for r_ind in remove_ind:
#                     remove_value.append(ss[r_ind])
#                 for v in remove_value:
#                     ss.remove(v)
#                 return ss

#         else:
#             return s
#     except:
#         print(s)
#         print()


def f1(r,p):
    f1 = None

    if r == 0 or p == 0:
        f1 = 0
    else:
        f1 = 2 * (p * r) / (p + r)
        f1 = round(f1,3)

    return f1
