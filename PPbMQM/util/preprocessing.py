import re
from nltk.tokenize import word_tokenize
def get_marked_text(target_sen):
    pattern = re.compile(r'<v>(.*?)</v>')
    marked_text = pattern.findall(target_sen)
    if len(marked_text) == 0:
        marked_text = ' '
    else:
        marked_text = marked_text[0]
    return marked_text


# def get_marked_text_index(target_sen):
#     ls = target_sen.split(' ')
#     span = {}
#     for ind, x in enumerate(ls):
#         if ["<", "v", ">"] == ls[ind: ind+3]:
#             span['start'] = ind
#         if ["<", "/", "v", ">"] == ls[ind: ind+4]:
#             span['end'] = ind
#     return str(span)

def get_marked_text_index_tokenize(target_sen):
    ls = word_tokenize(target_sen)
    span = {}
    for ind, x in enumerate(ls):
        if ["<", "v", ">"] == ls[ind: ind+3]:
            span['start'] = ind
        if ["<", "/v", ">"] == ls[ind: ind+3]:
            span['end'] = ind - 4
    return str(span)


def get_top_category(x):
    x = x.lower()
    result = x

    if 'accuracy' in x:
        result = 'Accuracy'
    if 'fluency' in x:
        result = 'Fluency'
    if 'terminology' in x:
        result = 'Terminology'
    if 'style' in x:
        result = 'Style'
    if 'locale convention' in x:
        result = 'Locale Convention'
    if 'non-translation' in x:
        result = 'Non-translation'
    return result


def get_omission_mqm(x,y,z):
    """
    Take x: category, y: severity, z: source as input
    output empty list if no omission error
    output a list of mqm annotations with all omission error
    """
    omission = 'No omission'
    if 'Omission' in x:
        omission = '_;_'.join([x,y,z])

    return omission


def get_error_list_no_source_issue(a):
    error_list_no_source_issue = []
    for x in a:
        x_split = x.split('_;_')
        if x_split[0] != 'Source issue':
            error_list_no_source_issue.append(x_split)
    return error_list_no_source_issue


import re
def get_mqm_score_from_meta(x):

    pre = 'metrics":{"MQM":'
    post = 'xxxxxx'
    x = x.replace(pre, post)

    pattern = r"(?<=xxxxxx)...."
    match = re.search(pattern, x)
    if match:
        result = match.group(0)
        return result
    else:
        return None


def extract_gold_error_list(gold):
    gold_list_1 = gold.split('---;---')
    gold_list = [error.split('_;_') for error in gold_list_1]

    result = []
    for error in gold_list:
        if error[0] == 'No-error':
            break #break and return an empty list for no error
        if error[0] == 'Source issue': # ignore source issue
            continue
        error_span = eval(error[3]) # in this case, error could be an accuracy/omission, while there is no marked text in the target
        if error_span == {}:
            continue

        result.append([error[0],error[1],error[2],error_span])
    # for each sub-list of error
    # position----  0: error type, 1: severity, 2: text, 3: index
    return result
