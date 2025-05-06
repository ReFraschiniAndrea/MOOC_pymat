__all__ = [
    "MatlabCode",
    "MatlabCodeBlock",
    "MatlabEnv",
    "MatlabCodeWithLogoo"
]

from manim import *
from Generic_mooc_utils import *
from custom_code import CustomCode, CodeWithLogo
from typing import Any

# import custom lexer and style for Colab-like python code listings
# Wanted to avoid to install the styles and lexers as plugins, but extremely hacky
from pygments.styles._mapping import STYLES

_MATLAB_LOGO = r'Assets\matlab_logo.png'
COLAB_FONT_SIZE = 12
MATLAB_FONT_SIZE = 12
MATLAB_PLOT_WIDTH = 2

class MatlabCode(CustomCode):
    # override default options
    default_background_config: dict[str, Any] = {
        "buff": 0.3,
        "fill_color": WHITE,
        "stroke_color": BLACK,
        "corner_radius": 0.2,
        "stroke_width": 0.5,
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

    def IntoMatlab(self, matlab_env: 'MatlabEnv', **kwargs):
        target = MatlabCodeBlock(self.code_string)
        matlab_env.add_cell(target)
        cells_to_fade = matlab_env.cells[:-1]
        if self.window is not None:
            return AnimationGroup(
                ReplacementTransform(self.window, target.window, **kwargs),
                ReplacementTransform(self.code, target.colabCode.code, **kwargs),
                # if we fade in the whole environment, also the new cell will appear before it should
                FadeIn(matlab_env.env_image, *cells_to_fade, target.gutter, target.playButton, **kwargs)
            )
        else:
            return AnimationGroup(
                ReplacementTransform(self.code, target.colabCode.code, **kwargs),
                FadeIn(matlab_env.env_image, *cells_to_fade, target.colabCode.window, target.gutter, target.playButton, **kwargs)
            )


class MatlabCodeBlock(MatlabCode):
    '''Utility for code to be displayed in the matlab environment.'''
    def __init__(self, code: str):
        super().__init__(code,
                         paragraph_config={'font_size': MATLAB_FONT_SIZE})
        target_height = self.colabCode.code.height + 2*_COLAB_SURROUND_CODE_BUFF if code != '' else 0.4766081925925926
        self.colabCode.add_background_window(
            Rectangle(
                width=_COLAB_BLOCK_WIDTH,
                height= target_height,
                fill_color=COLAB_LIGHTGRAY, 
                fill_opacity=1,
                stroke_width=0
            )
        )
        self.gutter = self._create_Gutter().next_to(self.colabCode.window, LEFT, buff=0)
        self.playButton = self._create_PlayButton().move_to(self.gutter.get_center()).align_to(self.gutter, UP).shift(DOWN*0.1)
        # be careful if the code is empty
        if self.colabCode.code.family_members_with_points():
            self.colabCode.code.align_to(self.colabCode.window, LEFT).shift(RIGHT*_COLAB_GUTTER_TO_TEXT_BUFF)
        self.output = None
        self.outputWindow = None

        self.add(self.colabCode)
        self.add(self.gutter)
        self.add(self.playButton)

class MatlabEnv(Mobject):
    def __init__(self, background=None):
        super().__init__()
        self.TOP_LEFT_CORNER_ = pixel2p(77, 156)
        self.OUTPUT_TOP_LEFT_CORNER_ = 0
        self.RUN_BUTTON_ = 0
        self.env_image = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-2)
        self.add(self.env_image)
        self.cells = []
        self.output_image=None
        self.output_text=None

    def set_image(self, image_path: str):
        self.env_image.become(ImageMobject(image_path).scale_to_fit_height(FRAME_HEIGHT)).set_z_index(-2)

    def add_cell(self, cell: MatlabCode):
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=0.1)
        self.cells.append(cell)
        self.add(cell)

    def remove_cell(self):
        if len(self.cells) > 0:
            self.remove(self.cells.pop())  
    
    def clear(self):
        while len(self.cells) > 0:
            self.remove_cell()
        if self.output_text is not None:
            self.remove(self.output_text)
            self.output_text = None
        if self.plot_window is not None:
            self.remove(self.plot_window)
            self.plot_window = None

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

    def add_output_text(self, output_text: str | Mobject):
        if isinstance(output_text, str):
            output_text = Paragraph(output_text, font_size=COLAB_FONT_SIZE,
                                    color=BLACK, font=CODE_FONT,
                                    line_spacing=0.5)
        output_text.move_to(self.OUTPUT_TOP_LEFT_CORNER_)
        self.add(output_text)

    def add_output_image(self, output_image: str | Mobject):
        if isinstance(output_image, str):
            self.output_image = ImageMobject(output_image)
        else:
            self.output_image = output_image
        self.output_image.scale_to_fit_width(MATLAB_PLOT_WIDTH).center()
        self.plot_window = SurroundingRectangle(output_image, color="#f7f7f7", buff=0.5, corner_radius=0.2,
                                                stroke_width=0.5, stroke_color=BLACK)
        self.add(self.plot_window, self.output_text)

    def focus_output(self, scale=0.75):
        
        if self.outputWindow is None or self.output is None:
            raise ValueError('Cell does not have output.')
        
        self.outputWindow.become(
                Rectangle(
                color=WHITE,
                height=FRAME_HEIGHT,
                width=FRAME_WIDTH,
                fill_opacity=1)
            )
        self.output.scale_to_fit_width(FRAME_WIDTH*scale).center()


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