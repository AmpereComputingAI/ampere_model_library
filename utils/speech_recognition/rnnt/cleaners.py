# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2017 Keith Ito
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" from https://github.com/keithito/tacotron
Modified to add punctuation removal
"""

'''
Cleaners are transformations that run over the input text at both training and eval time.
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def remove_punctuation(text, table):
    text = text.translate(table)
    text = re.sub(r'&', " and ", text)
    text = re.sub(r'\+', " plus ", text)
    return text

def english_cleaners(text, table=None):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = unidecode(text)
    text = text.lower()
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    if table is not None:
        text = remove_punctuation(text, table)
    text = re.sub(_whitespace_re, ' ', text)
    return text
