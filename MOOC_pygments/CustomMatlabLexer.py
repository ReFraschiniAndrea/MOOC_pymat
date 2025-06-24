"""z

    Lexers for Matlab and related languages.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import Lexer, RegexLexer, bygroups, default, words, \
    do_insertions, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Generic, Whitespace

from pygments.lexers import _scilab_builtins

__all__ = ['CustomMatlabLexer']


class CustomMatlabLexer(RegexLexer):
    """
    For Matlab source code.
    """
    name = 'Custom-matlab'
    aliases = ['custommatlab']
    filenames = ['*.m']
    mimetypes = ['text/matlab']
    url = 'https://www.mathworks.com/products/matlab.html'
    version_added = '0.10'

    _operators = r'-|==|~=|<=|>=|<|>|&&|&|~|\|\|?|\.\*|\*|\+|\.\^|\^|\.\\|\./|/|\\'

    tokens = {
        'expressions': [
            # operators:
            (_operators, Operator),

            # numbers (must come before punctuation to handle `.5`; cannot use
            # `\b` due to e.g. `5. + .5`).  The negative lookahead on operators
            # avoids including the dot in `1./x` (the dot is part of `./`).
            (rf'(?<!\w)((\d+\.\d+)|(\d*\.\d+)|(\d+\.(?!{_operators})))'
             r'([eEf][+-]?\d+)?(?!\w)', Number.Float),
            (r'\b\d+[eEf][+-]?[0-9]+\b', Number.Float),
            (r'\b\d+\b', Number.Integer),

            # punctuation:
            (r'\[|\]|\(|\)|\{|\}|:|@|\.|,', Punctuation),
            (r'=|:|;', Punctuation),

            # quote can be transpose, instead of string:
            # (not great, but handles common cases...)
            (r'(?<=[\w)\].])\'+', Operator),

            (r'"(""|[^"])*"', String),

            (r'(?<![\w)\].])\'', String, 'string'),
            (r'[a-zA-Z_]\w*', Name),
            (r'\s+', Whitespace),
            (r'.', Text),
        ],
        'root': [
            # line starting with '!' is sent as a system command.  not sure what
            # label to use...
            (r'^!.*', String.Other),
            (r'%\{\s*\n', Comment.Multiline, 'blockcomment'),
            (r'%.*$', Comment),
            (r'(\s*^\s*)(function)\b', bygroups(Whitespace, Keyword), 'deffunc'),
            (r'(\s*^\s*)(properties)(\s+)(\()',
             bygroups(Whitespace, Keyword, Whitespace, Punctuation),
             ('defprops', 'propattrs')),
            (r'(\s*^\s*)(properties)\b',
             bygroups(Whitespace, Keyword), 'defprops'),

            # from 'iskeyword' on version 9.4 (R2018a):
            # Check that there is no preceding dot, as keywords are valid field
            # names.
            (words(('break', 'case', 'catch', 'classdef', 'continue',
                    'dynamicprops', 'else', 'elseif', 'end', 'for', 'function',
                    'global', 'if', 'methods', 'otherwise', 'parfor',
                    'persistent', 'return', 'spmd', 'switch',
                    'try', 'while'),
                   prefix=r'(?<!\.)(\s*)(', suffix=r')\b'),
             bygroups(Whitespace, Keyword)),

            (
                words(
                    [
                        # Normally a list of all Matlab built-in functions (>2000)
                        # I'm going to just add manually the few that are used in the mooc.
                        "addpath",
                        "fprintf",
                        "cos",
                        "sin",
                    ],
                    prefix=r"(?<!\.)(",  # Exclude field names
                    suffix=r")\b"
                ),
                Name.Builtin
            ),

            # line continuation with following comment:
            (r'(\.\.\.)(.*)$', bygroups(Keyword, Comment)),

            # command form:
            # "How MATLAB Recognizes Command Syntax" specifies that an operator
            # is recognized if it is either surrounded by spaces or by no
            # spaces on both sides (this allows distinguishing `cd ./foo` from
            # `cd ./ foo`.).  Here, the regex checks that the first word in the
            # line is not followed by <spaces> and then
            # (equal | open-parenthesis | <operator><space> | <space>).
            (rf'(?:^|(?<=;))(\s*)(\w+)(\s+)(?!=|\(|{_operators}\s|\s)',
             bygroups(Whitespace, Name, Whitespace), 'commandargs'),

            include('expressions')
        ],
        'blockcomment': [
            (r'^\s*%\}', Comment.Multiline, '#pop'),
            (r'^.*\n', Comment.Multiline),
            (r'.', Comment.Multiline),
        ],
        'deffunc': [
            (r'(\s*)(?:(\S+)(\s*)(=)(\s*))?(.+)(\()(.*)(\))(\s*)',
             bygroups(Whitespace, Name, Whitespace, Punctuation,
                      Whitespace, Name.Function, Punctuation, Name,
                      Punctuation, Whitespace), '#pop'),
            # function with no args
            (r'(\s*)([a-zA-Z_]\w*)',
             bygroups(Whitespace, Name.Function), '#pop'),
        ],
        'propattrs': [
            (r'(\w+)(\s*)(=)(\s*)(\d+)',
             bygroups(Name.Builtin, Whitespace, Punctuation, Whitespace,
                      Number)),
            (r'(\w+)(\s*)(=)(\s*)([a-zA-Z]\w*)',
             bygroups(Name.Builtin, Whitespace, Punctuation, Whitespace,
                      Keyword)),
            (r',', Punctuation),
            (r'\)', Punctuation, '#pop'),
            (r'\s+', Whitespace),
            (r'.', Text),
        ],
        'defprops': [
            (r'%\{\s*\n', Comment.Multiline, 'blockcomment'),
            (r'%.*$', Comment),
            (r'(?<!\.)end\b', Keyword, '#pop'),
            include('expressions'),
        ],
        'string': [
            (r"[^']*'", String, '#pop'),
        ],
        'commandargs': [
            # If an equal sign or other operator is encountered, this
            # isn't a command. It might be a variable assignment or
            # comparison operation with multiple spaces before the
            # equal sign or operator
            (r"=", Punctuation, '#pop'),
            (_operators, Operator, '#pop'),
            (r"[ \t]+", Whitespace),
            ("'[^']*'", String),
            (r"[^';\s]+", String),
            (";", Punctuation, '#pop'),
            default('#pop'),
        ]
    }

    def analyse_text(text):
        # function declaration.
        first_non_comment = next((line for line in text.splitlines()
                                  if not re.match(r'^\s*%', text)), '').strip()
        if (first_non_comment.startswith('function')
                and '{' not in first_non_comment):
            return 1.
        # comment
        elif re.search(r'^\s*%', text, re.M):
            return 0.2
        # system cmd
        elif re.search(r'^!\w+', text, re.M):
            return 0.2

line_re  = re.compile('.*?\n')
