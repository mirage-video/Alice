import html

import regex as re

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
