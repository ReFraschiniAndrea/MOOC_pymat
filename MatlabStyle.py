"""
    Matlab like style.

    Style that emulates Matlab's default editor look.
"""

# from pygments.style import Style
# from pygments.token import Keyword, Name, Comment, String, Error, \
#     Number, Other, Whitespace, Generic

from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
    Number, Operator, Generic, Whitespace, Punctuation, Other, Literal

__all__ = ['MatlabStyle']

# Matlab editor style (colors taken from the official documentation)
WHITE = "#FFFFFF"
BLACK = "#000000"
GREEN = "#028008"
PURPLE = "#ab04f9"
BLUE = "#0d00ff"
RED = "#FF0000"

class MatlabStyle(Style):
    name = 'matlab'
    
    background_color = WHITE

    styles = {
        Text:                      BLACK,
        Whitespace:                "",
        Error:                     RED,
        Other:                     "",

        Comment:                   GREEN,
        Keyword:                   BLUE,

        Operator:                  BLACK,      # +,-,<,...
        Operator.Word:             BLACK,      # in, and, or
        Punctuation:               BLACK,
        Name:                      BLACK,

        Number:                    BLACK,
        String:                    PURPLE,
}