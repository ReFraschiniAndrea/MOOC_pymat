"""
    Colab like style.

    Style for pyhton code imitating Google Colab's appearence.
"""

from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
    Number, Operator, Generic, Whitespace, Punctuation, Other, Literal


__all__ = ['ColabStyle']

# Google Colab style
BLACK = "#000000"
LIGHTGRAY = "#f7f7f7"
GREEN = "#007900"
TEAL = "#257693"
PINE = "#116644"
PURPLE = "#cf70e7"
BLUE = "#0431fa"
DEEPBLUE = "#001080"
BROWN = "#795e26"
CRIMSON = "#a31515"
RED = "#FF0000"

class ColabStyle(Style):
    name = 'colab'
    
    background_color = LIGHTGRAY

    styles = {
        Text:                      BLACK,
        Whitespace:                "",
        Error:                     RED,
        Other:                     "",

        Comment:                   GREEN,

        Keyword:                   PURPLE,
        Keyword.Constant:          "",
        Keyword.Declaration:       BLUE,
        Keyword.Namespace:         PURPLE,      # from, import
        Keyword.Pseudo:            "",  
        Keyword.Reserved:          "",  
        Keyword.Type:              TEAL,        # int
        Keyword.Definition:        BLUE,
        Operator:                  BLACK,       # +,-,<,...
        Operator.Word:             BLUE,        # in, and, or

        Punctuation:               BLACK,       # :,.: etc.
        Punctuation.Parenthesis:   BLUE,        # CUSTOM: (), [], {}

        Name:                      BLACK,
        Name.Attribute:            BLUE,
        Name.Builtin:              BROWN,
        Name.Builtin.Pseudo:       "",
        Name.Class:                TEAL,        # class names
        Name.Constant:             RED,
        Name.Decorator:            "",
        Name.Entity:               "",
        Name.Exception:            RED,
        Name.Function:             BROWN,       # function names
        Name.Property:             "",
        Name.Label:                "",
        Name.Namespace:            BLACK,       # e.g. numpy, np
        Name.Other:                BLUE,
        Name.Tag:                  "",
       
        Number:                    PINE,        # number literals

        Literal:                   RED,
        Literal.Date:              GREEN,

        String:                    CRIMSON,     # strings like "hello"
        String.Backtick:           "",
        String.Char:               BLACK,
        String.Doc:                GREEN,
        String.Double:             "",
        String.Escape:             RED,
}