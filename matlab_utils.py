__all__ = [
    "COLAB_LIGHTGRAY",
    "COLAB_GRAY",
    "COLAB_DARKGRAY",
    "COLAB_FONT_SIZE",
    "ColabCode",
    "ColabCodeBlock",
    "ColabBlockOutputText",
    "ColabEnv"
]

from manim import *
from Generic_mooc_utils import *
from custom_code import CustomCode, CodeWithLogo
from typing import Any

# import custom lexer and style for Colab-like python code listings
# Wanted to avoid to install the styles and lexers as plugins, but extremely hacky
from pygments.styles._mapping import STYLES

_MATLAB_LOGO = r'Assets\matlab_logo.png'


class MatlabCode(CustomCode):
    # override default options
    default_background_config: dict[str, Any] = {
        "buff": 0.3,
        "fill_color": WHITE,
        "stroke_color": BLACK,
        "corner_radius": 0.2, # sharp corners
        "stroke_width": 0.5,  # no outline
        "fill_opacity": 1, 
    }
    default_paragraph_config: dict[str, Any] = {
        "font": CODE_FONT,
        "font_size": 24,
        "line_spacing": 0.5,
        "disable_ligatures": True,
    }
    def __init__(
        self,
        code_string: str = None, 
        code_file = None,
        paragraph_config = None
    ):
        super().__init__(
            code_file, 
            code_string, 
            language='matlab',
            formatter_style='matlab',
            paragraph_config=paragraph_config
        )

    def IntoMatlab(self, matlab_env, **kwargs):
      pass


# class MatlabBlockOutput(Paragraph):
#     def __init__(self, text, **kwargs):
#         super().__init__(text, font_size=COLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
#                         line_spacing=0.5, **kwargs)

class MatlabEnv(Mobject):
    def __init__(self, background=None):
        super().__init__()
        self.TOP_LEFT_CORNER_ = pixel2p(77, 156)
        self.OUTPUT_TOP_LEFT_CORNER_ = 0
        self.env_image = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-2)
        self.add(self.env_image)
        self.cells = []

    def set_image(self, image_path: str):
         self.env_image.become(ImageMobject(image_path).scale_to_fit_height(FRAME_HEIGHT)).set_z_index(-2)

    # def add_cell(self, cell: ColabCodeBlock):
    #     if len(self.cells) == 0:
    #         cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
    #     else:
    #         cell.next_to(self.cells[-1], DOWN, buff=_COLAB_SURROUND_CODE_BUFF)
    #     self.cells.append(cell)
    #     self.add(cell)

    # def remove_cell(self):
    #     if len(self.cells) > 0:
    #         self.remove(self.cells.pop())  
    
    # def clear(self):
    #     while len(self.cells) > 0:
    #         self.remove_cell()

    # def OutofMatlab(self, cell: ColabCodeBlock, fullscreen=True, **kwargs):
    #     target = ColabCode(cell.colabCode.code_string)
    #     target.add_background_window()
    #     if fullscreen:
    #         target.window.become(
    #             Rectangle(
    #                 fill_color=COLAB_LIGHTGRAY,
    #                 fill_opacity=1,
    #                 height=FRAME_HEIGHT,
    #                 width=FRAME_WIDTH))
    #     return AnimationGroup(
    #         Transform(cell.colabCode, target),
    #         FadeOut(self.env_image, cell.gutter, cell.playButton)
    # )

class MatlabCodeWithLogo(CodeWithLogo):
    def __init__(
        self,
        code,
        logo_pos=UP,
        logo_buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        **kwargs
    ):
        super().__init__(
            code_mobj=MatlabCode(code, **kwargs),
            logo_mobj=ImageMobject(_MATLAB_LOGO).scale(0.5),
            logo_pos=logo_pos,
            logo_buff=logo_buff,
        )