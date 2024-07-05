def mqm_scoring(mqms):
    x = []
    score = 0
    if len(mqms) == 0:
        return 0
    else:
        score = 0
        for mqm in mqms:
            x.append([mqm[0].lower(), mqm[1].lower()]) # error type, severity
        for xx in x:
            if xx[1] == 'major':
                score += 5
            elif xx[1] == 'minor':
                score += 1
                        # elif xx[1] == 'minor' and xx[0]== 'fluency': # for minor fluency errors
            #     score += 0.1
        return score


def gold_list_omission(x,y):
    for xx in x.split('---;---'):
        if xx != 'No omission':
            omission_error = xx.split('_;_')
            omission_error[0] = 'Omission'
            sen = omission_error[2]
            omission_error[2] = get_marked_text(sen)
            omission_error.append(get_marked_text_index_tokenize(sen))
            y.append(omission_error)
    return y

import re
from nltk.tokenize import word_tokenize
import jieba
def get_marked_text(target_sen):
    pattern = re.compile(r'<v>(.*?)</v>')
    marked_text = pattern.findall(target_sen)
    if len(marked_text) == 0:
        marked_text = ' '
    else:
        marked_text = marked_text[0]
    return marked_text


def get_marked_text_index_tokenize(target_sen):
    ls = list(jieba.cut(target_sen))
    span = {}
    for ind, x in enumerate(ls):
        if ["<", "v", ">"] == ls[ind: ind+3]:
            span['start'] = ind + 3
        if ["<","/" ,"v", ">"] == ls[ind: ind+4]:
            span['end'] = ind -1
    return span
