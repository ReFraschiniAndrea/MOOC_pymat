"""Custom mod of Code mobject.
Mainly to allow for not having a background by default"""

from __future__ import annotations

__all__ = [
    "CustomCode",
]

from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer, guess_lexer_for_filename

from manim.constants import *
from manim.mobject.geometry.polygram import Rectangle
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.typing import StrPath
from manim.utils.color import WHITE, ManimColor


class CustomCode(VMobject):
    """A highlighted source code listing."""

    _styles_list_cache: list[str] | None = None
    default_background_config: dict[str, Any] = {
        "buff": 0.3,
        "fill_color": ManimColor("#222"),
        "stroke_color": WHITE,
        "corner_radius": 0.2,
        "stroke_width": 1,
        "fill_opacity": 1,
    }
    default_paragraph_config: dict[str, Any] = {
        "font": "Monospace",
        "font_size": 24,
        "line_spacing": 0.5,
        "disable_ligatures": True,
    }

    def __init__(
        self,
        code_file: StrPath | None = None,
        code_string: str | None = None,
        language: str | None = None,
        formatter_style: str = "vim",
        tab_width: int = 4,
        paragraph_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        # get the code string
        if code_file is not None:
            code_file = Path(code_file)
            code_string = code_file.read_text(encoding="utf-8")
            lexer = guess_lexer_for_filename(code_file.name, code_string)
        elif code_string is not None:
            if language is not None:
                lexer = get_lexer_by_name(language)
            else:
                lexer = guess_lexer(code_string)
        else:
            raise ValueError("Either a code file or a code string must be specified.")

        code_string = code_string.expandtabs(tabsize=tab_width)
        self.code_string = code_string

        # Create Paragraph object corresponding to the code
        formatter = HtmlFormatter(
            style=formatter_style,
            noclasses=True,
            cssclasses="",
        )
        soup = BeautifulSoup(
            highlight(code_string, lexer, formatter), features="html.parser"
        )
        self._code_html = soup.find("pre")
        assert isinstance(self._code_html, Tag)

        # as we are using Paragraph to render the text, we need to find the character indices
        # of the segments of changed color in the HTML code
        color_ranges = []
        current_line_color_ranges = []
        current_line_char_index = 0
        for child in self._code_html.children:
            if child.name == "span":
                try:
                    child_style = child["style"]
                    if isinstance(child_style, str):
                        color = child_style.removeprefix("color: ")
                    else:
                        color = None
                except KeyError:
                    color = None
                current_line_color_ranges.append(
                    (
                        current_line_char_index,
                        current_line_char_index + len(child.text),
                        color,
                    )
                )
                current_line_char_index += len(child.text)
            else:
                for char in child.text:
                    if char == "\n":
                        color_ranges.append(current_line_color_ranges)
                        current_line_color_ranges = []
                        current_line_char_index = 0
                    else:
                        current_line_char_index += 1

        color_ranges.append(current_line_color_ranges)
        code_lines = self._code_html.get_text().removesuffix("\n").split("\n")

        if paragraph_config is None:
            paragraph_config = {}
        base_paragraph_config = self.default_paragraph_config.copy()
        base_paragraph_config.update(paragraph_config)

        self.code = Paragraph(
            *code_lines,
            **base_paragraph_config,
        )
        for line, color_range in zip(self.code, color_ranges):
            for start, end, color in color_range:
                line[start:end].set_color(color)

        self.add(self.code)


    def __getitem__(self, value):
        return self.code.__getitem__(value)


    def add_background_window(self, rectangle: Rectangle = None, background_config=None):
        if rectangle is not None:
            self.window = rectangle
        else:
            # default background window
            if background_config is None:
                background_config = {}
            background_config_base = self.default_background_config.copy()
            background_config_base.update(background_config)

            self.window = SurroundingRectangle(
                self,
                **background_config_base,
            )

        self.add_to_back(self.window)

    def TypeLetterbyLetter(self, lines: int =None, lag_ratio=1):
        if lines is None:
            lines = range(len(self.code))
        anims = []
        for i in lines:
            if self.code[i].family_members_with_points():  # Not an empty line
                anims.append(AddTextLetterByLetter(self.code[i], rate_func=linear, time_per_char=0.01))
        return AnimationGroup(*anims, lag_ratio=lag_ratio)

