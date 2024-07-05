import pandas as pd
import ast
from nltk import word_tokenize
import matplotlib.pyplot as plt
def calculate_quality_score(mqm,target):
    quality_score = 1
    target = word_tokenize(target)

    if len(mqm) == 0:
        return quality_score
    else:
        try:
            etpt = 0
            for error in mqm:
                if error[1].lower() == 'major':
                    etpt += 5
                if error[1].lower() == 'minor':
                    etpt += 1
            pwpt = etpt / len(target)
            quality_score = 1 - pwpt
            if quality_score < 0:
                quality_score = 0
        except:
            quality_score = None


    return quality_score



def get_target_sen_length(target_sen):
    ls = word_tokenize(target_sen)
    return len(ls)
