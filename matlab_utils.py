__all__ = [
    "MatlabCode",
    "MatlabCodeBlock",
    "MatlabEnv",
    "MatlabOutputText",
    "MatlabCodeWithLogo"
]

from manim import *
from Generic_mooc_utils import *
from custom_code import CustomCode, CodeWithLogo
from typing import Any

# import custom lexer and style for Colab-like python code listings
# Wanted to avoid to install the styles and lexers as plugins, but extremely hacky
from pygments.styles._mapping import STYLES
from pygments.lexers._mapping import LEXERS

# STYLES['ColabStyle'] = ('ColabStyle', 'colab', ())
# _STYLE_NAME_TO_MODULE_MAP['colab'] = 'ColabStyle'
# pygments.styles.STYLES['colab'] = ColabStyle  # Optional for some versions
LEXERS['CustomMatlabLexer'] = (
    'CustomMatlabLexer', # name of the module
    'Custom-matlab', # Name of the lexer
    ('custommatlab',), # aliases
    ('*.m',), # extensions
    () # mime types
    )

_MATLAB_LOGO = r'Assets\matlab_logo.png'
COLAB_FONT_SIZE = 12
MATLAB_FONT_SIZE = 12
MATLAB_PLOT_WIDTH = 5
_MATLAB_CELL_TO_CELL_BUFF = 0.25
MATLAB_GRAY = "#f0f0f0"

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
            language='custommatlab',
            formatter_style='matlab',
            paragraph_config=paragraph_config
        )

    def IntoMatlab(self, matlab_env: 'MatlabEnv', **kwargs):
        target = MatlabCodeBlock(self.code_string)
        # target.add_background_window(background_config={'stroke_width':0, 'buff':0})
        matlab_env.add_cell(target)
        cells_to_fade = matlab_env.cells[:-1]
        if self.window is not None:
            fake_dot = Dot(fill_opacity=0)
            return AnimationGroup(
                ReplacementTransform(self.window, target.window, **kwargs),
                ReplacementTransform(self.code, target.code, **kwargs),
                # if we fade in the whole environment, also the new cell will appear before it should
                # Issue #3039: env_image must not be already added OR the only thing faded in.
                # We add a transparent dot as an ugly hack to circumvent it.
                FadeIn(matlab_env.env_image, *cells_to_fade, fake_dot, **kwargs),
                FadeOut(fake_dot)
        )
        else:
            return AnimationGroup(
                ReplacementTransform(self.code, target.code, **kwargs),
                FadeIn(matlab_env.env_image, *cells_to_fade, target.window, **kwargs)
            )


class MatlabCodeBlock(MatlabCode):
    '''Utility for code to be displayed in the matlab environment.'''
    def __init__(self, code: str):
        super().__init__(code,
                         paragraph_config={'font_size': MATLAB_FONT_SIZE})
        if code == '':  # empty cell
            self.add_background_window(Rectangle(
                color=WHITE,
                height=(740-240)/1050*FRAME_HEIGHT,
                width=(815-144)/1400*FRAME_WIDTH,
                stroke_width=0,
                fill_opacity=1
            ))
        else:
            self.add_background_window(background_config={'stroke_color':WHITE, 'stroke_width':0, 'corner_radius':0, 'buff':0})


class MatlabEnv(Mobject):
    def _pixel2p(px, py):
        '''Converts pixel coordinates (1080 x 1440) into manim units.'''
        return [
             (px/1400 - 0.5)*FRAME_WIDTH,
            -(py/1050 - 0.5)*FRAME_HEIGHT,
            0
        ]
    
    TOP_LEFT_CORNER_ = _pixel2p(155, 241)
    TOP_LEFT_CORNER_UNSAVED_ = _pixel2p(128, 218)
    OUTPUT_TOP_LEFT_CORNER_ = _pixel2p(80, 783)
    NEW_SCRIPT_ = _pixel2p(25, 107)
    SAVE_ = _pixel2p(123, 71)
    SAVE_PROMPT_BUTTON_ = _pixel2p(874, 888)
    RUN_BUTTON_ = _pixel2p(1070, 67)

    def __init__(self, background=None):
        super().__init__()
        
        self.env_image = ImageMobject(background).scale_to_fit_width(FRAME_WIDTH).center().set_z_index(-2)
        self.add(self.env_image)
        self.cells = []
        self.output : MatlabOutput  | None = None
        self.cursor : Cursor = Cursor().set_z_index(-1.5)


    def set_image(self, image_path: str):
        self.env_image.become(ImageMobject(image_path).scale_to_fit_width(FRAME_WIDTH)).center().set_z_index(-2)

    def add_cell(self, cell: MatlabCodeBlock):
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=_MATLAB_CELL_TO_CELL_BUFF).align_to(self.cells[-1], LEFT)
        cell.set_z_index(-1.5)
        self.cells.append(cell)
        self.add(cell)

    def remove_cell(self):
        if len(self.cells) > 0:
            self.remove(self.cells.pop())  
    
    def clear(self):
        while len(self.cells) > 0:
            self.remove_cell()
        if self.output is not None:
            self.remove(self.output)
            self.output = None

    def OutofMatlab(self, cell: MatlabCodeBlock, fullscreen=True, **kwargs):
        target = MatlabCode(cell.code_string)
        target.add_background_window()
        if fullscreen:
            target.window.become(
                Rectangle(
                    fill_color=WHITE,
                    fill_opacity=1,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH))
        return AnimationGroup(
            Transform(cell, target),
            FadeOut(self.env_image)
    )

    def add_output(
        self,
        output_text: str | Mobject = None,
        output_image: str | Mobject = None
    ):
        # clear previous output
        if self.output is not None:
            self.remove(self.output)
        self.output = MatlabOutput(output_text, output_image).set_z_index(-1.5)
        self.add(self.output)
        
    def Run(self):
        self.cursor.move_to(self.RUN_BUTTON_)
        self.add(self.cursor)
        if self.output is not None:
        # add again the output so it is above everything else
            # self.remove(self.output)
            # self.add(self.output)
            return Succession(
                GrowFromCenter(self.cursor),
                self.cursor.Click(),
                FadeIn(self.output, run_time=0),
                Wait(0.1)
            )
        else:
            return Succession(GrowFromCenter(self.cursor), self.cursor.Click())

    def focus_output(self, scale=0.75, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2, **kwargs):
        if self.output is None:
            raise ValueError("Matlab environment has no output")
        self.output.set_z_index(0)  # any negative z_index will not work
        return self.output.animate(**kwargs).focus(scale, buff)


class MatlabOutputText(Paragraph):
    def __init__(self, text, **kwargs):
        super().__init__(text, font_size=COLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
                        line_spacing=0.5, **kwargs)
        
class MatlabOutput(Mobject):
    def __init__(
        self,
        output_text: str | Mobject = None,
        output_image: str | Mobject = None
    ):
        super().__init__()
        # add output text
        if output_text is None:
            self.text = None
        else:
            if isinstance(output_text, str):
                self.text = MatlabOutputText(output_text)
            else:
                self.text=output_text
            self.text.move_to(MatlabEnv.OUTPUT_TOP_LEFT_CORNER_, aligned_edge=UL)
            # add background output window
            self.outputWindow = SurroundingRectangle(self.text, color=WHITE, corner_radius=0, fill_opacity=1, buff=0)
            self.add(self.outputWindow, self.text)

        # add output image
        if output_image is None:
            self.image = None
        else:
            if isinstance(output_image, str):
                self.image = ImageMobject(output_image)
            else:
                self.image = output_image
            self.image.scale_to_fit_width(MATLAB_PLOT_WIDTH).center()
            self.plot_window = SurroundingRectangle(self.image, color="#f0f0f0", 
                                                    buff=0.1, corner_radius=0.1,
                                                    fill_opacity=1,
                                                    stroke_width=0.5, stroke_color=BLACK)
            self.add(self.plot_window, self.image)

    def focus(self, scale=0.75, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2):
        """To be used with animate syntax"""
        self.text.scale_to_fit_width(FRAME_WIDTH*scale)
        if self.image is not None:
            Group(self.image, self.plot_window).next_to(self.text, DOWN, buff=buff)
        self.center()
        if self.outputWindow is not None:
            self.outputWindow.become(
                    Rectangle(
                    color=WHITE,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH,
                    fill_opacity=1)
                )

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