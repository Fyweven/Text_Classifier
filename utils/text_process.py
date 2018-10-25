# -*- coding: utf-8 -*-
import codecs
import sys
sys.path.append('..')
from utils.langconv import *

translate_to = u' '
not_letters_or_digits = u'.!"#%\'()*+,-:;<=>?@[\]^_`{|}~┫┃┗┻┛％［「丶¯〓☞＝☆•↑～．↓「」★□－→・※“”：！【】·？《》，… ╗╚┐└—。、（）；/’‘'
translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)

def format_char(to_translate):
    input_data = to_translate
    output = input_data.translate(translate_table)
    return output

def _format_tt(text):
    """去除标点符号"""
    us = list(text.strip())
    us = [ format_char(v) for v in us ]
    us = [ u for u in us if ' ' not in u ]
    us = [ u.lower() for u in us ]
    return "".join(us)

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def Simplified2Traditional(sentence):
    '''
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    '''
    sentence = Converter('zh-hant').convert(sentence)
    return sentence

def _to_unicode(v):
    u = v
    if isinstance(v, str):
        u = v.decode('utf-8')
    return u

def Q2B(uchar):
    """unicode, 全角转半角"""
    inside_code=ord(uchar)
    if inside_code==0x3000:
        inside_code=0x0020
    else:
        inside_code-=0xfee0
    if inside_code<0x0020 or inside_code>0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):
    """unicode, 把字符串全角转半角"""
    try:
        return "".join([Q2B(uchar) for uchar in ustring])
    except:
        return ustring

def uniform(ustring):
    """格式化字符串，繁体转简体，完成全角转半角，大写转小写的工作"""
    ustring = _to_unicode(ustring)
    ustring = Traditional2Simplified(ustring)
    text = stringQ2B(ustring).lower()
    text = _format_tt(text)
    return text

if __name__ == '__main__':
    tt = '海 南 将 建 自 贸 区 自 贸 港 政 策 利 好 三 亚 免 税 # 一 * 行 '
    print uniform(tt)
