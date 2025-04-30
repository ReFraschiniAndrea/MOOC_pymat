__all__ = [
    "COLAB_LIGHTGRAY",
    "COLAB_GRAY",
    "COLAB_DARKGRAY",
    "COLAB_FONT_SIZE",

]

from manim import *
from Generic_mooc_utils import *
from custom_code import CustomCode
from typing import Any

# Colab constants
COLAB_LIGHTGRAY = "#f7f7f7"
COLAB_GRAY = "#eeeeee"
COLAB_DARKGRAY ="#424242"
COLAB_FONT_SIZE = 12
_COLAB_LEFT_BUFF = 127/1440 * FRAME_WIDTH
_COLAB_GUTTER_WIDTH = 50/1080 *FRAME_HEIGHT
_COLAB_BLOCK_WIDTH = 1300/1440 * FRAME_WIDTH
_COLAB_BUTTON_RADIUS = (23/1080*FRAME_HEIGHT)/2
_COLAB_SURROUND_CODE_BUFF = 17/1080*FRAME_HEIGHT
_COLAB_GUTTER_TO_TEXT_BUFF = 10/1440 * FRAME_WIDTH
_PYTHON_LOGO = r'Assets\python_logo.png'


class ColabCode(CustomCode):
    # override default options
    default_background_config: dict[str, Any] = {
        "fill_color": COLAB_LIGHTGRAY,
        "stroke_color": BLACK,
        "corner_radius": 0.2,
        "stroke_width": 0,  # no outline
        "fill_opacity": 1, 
        "buff": MED_SMALL_BUFF
    }
    default_paragraph_config: dict[str, Any] = {
        "font": "Monospace",
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
            language='python',
            formatter_style='paraiso-light',
            paragraph_config=paragraph_config
        )
    # def __init__(self, code, include_logo=False, logo_pos=UP, logo_buff =DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    #              font_size=24, line_spacing=0.5, corner_radius=0.2, fill_opacity=1, bckgr_buff=MED_SMALL_BUFF):
    #     if include_logo:
    #         logo = ImageMobject(_PYTHON_LOGO).scale(0.5)
    #     else:
    #         logo = None
    #     super().__init__(
    #         code_string=code,
    #         language='python',
    #         formatter_style='paraiso-light',
    #         include_logo=include_logo,
    #         logo_mobj=logo,
    #         logo_buff=logo_buff,
    #         logo_pos=logo_pos,
    #         background_config={
    #             "buff": 0.3,
    #             "fill_color": COLAB_LIGHTGRAY,
    #             "stroke_color": BLACK,
    #             "corner_radius":corner_radius, # sharp corners
    #             "stroke_width": 0,  # no outline
    #             "fill_opacity": fill_opacity, 
    #             "buff": bckgr_buff
    #         },
    #         paragraph_config={
    #             "font": CODE_FONT,
    #             "font_size": font_size,
    #             "line_spacing": line_spacing,
    #             "disable_ligatures": True,
    #             "color":BLACK   # color of the line numbers
    #         }
    #     )

    def into_colab(self, colab_env: 'ColabEnv', **kwargs):
        target = ColabCodeBlock(self.code_string)
        colab_env.add_cell(target)
        cells_to_fade = colab_env.cells[:-1]
        return AnimationGroup(
            ReplacementTransform(self.window, target.colabCode.window, **kwargs),
            ReplacementTransform(self.code, target.colabCode.code, **kwargs),
            # if we fade in the whole environment, also the new cell will appear before it should
            FadeIn(colab_env.env_image, *cells_to_fade, target.gutter, target.playButton, **kwargs)
        )


class ColabCodeBlock(Mobject):
    def __init__(self, code: str):
        super().__init__()
        self.colabCode = ColabCode(code,
                         font_size=COLAB_FONT_SIZE,
                         line_spacing=0.5, corner_radius=0, fill_opacity=1,
                        )
        self.colabCode.add_background_window(
            Rectangle(
                width=_COLAB_BLOCK_WIDTH,
                height= 0 if code != '' else 0.4766081925925926,
                fill_color=COLAB_LIGHTGRAY, 
                fill_opacity=1,
                stroke_width=0
            )
        )
        self.gutter = self._create_Gutter().next_to(self.colabCode.window, LEFT, buff=0)
        self.playButton = self._create_PlayButton().move_to(self.gutter.get_center()).align_to(self.gutter, UP).shift(DOWN*0.1)
        # be careful if empty
        if code != '':
            self.colabCode.code.align_to(self.colabCode.window, LEFT).shift(RIGHT*_COLAB_GUTTER_TO_TEXT_BUFF)
        self.output = None
        self.outputWindow = None

        self.add(self.colabCode)
        self.add(self.gutter)
        self.add(self.playButton)

    def _create_PlayButton(self):
        button = Circle(
            color=COLAB_DARKGRAY, 
            radius=_COLAB_BUTTON_RADIUS,
            fill_opacity=1)
        tri = Triangle(
            color=COLAB_LIGHTGRAY,
            radius=_COLAB_BUTTON_RADIUS*0.7,
            fill_opacity=1,
            stroke_width=0
            ).rotate(-PI/2).move_to(button).shift(RIGHT*_COLAB_BUTTON_RADIUS*0.7/4)
        return VGroup(button, tri)
    
    def _create_Gutter(self):
        return Rectangle(
            color=COLAB_GRAY, 
            width=_COLAB_GUTTER_WIDTH, 
            height=self.colabCode.window.height,
            fill_opacity=1,
            stroke_width=0)
        
    def add_output(self, output: str | Mobject):
        if isinstance(output, str):
            self.output = ColabBlockOutputText(output)
        else:
            self.output=output
        self.outputWindow = Rectangle(
            width=_COLAB_BLOCK_WIDTH + _COLAB_GUTTER_WIDTH,
            height=self.output.height + 2 * _COLAB_SURROUND_CODE_BUFF,
            color=WHITE,
            fill_opacity=1,
            stroke_width=0,
        )
        self.output.move_to(self.outputWindow)
        Group(self.outputWindow, self.output).next_to(VGroup(self.colabCode.window, self.gutter), DOWN, buff=0)
        self.output.align_to(self.colabCode.code, LEFT)

        self.add(self.outputWindow)
        self.add(self.output)

    def run(self):
        self.cursor = Cursor().move_to(self.playButton)
        self.add(self.cursor)
        return Succession(
            GrowFromCenter(self.cursor),
            AnimationGroup(
                self.cursor.click(),
                FadeIn(self.outputWindow, self.output, run_time=0),
                lag_ratio=0.5),
            lag_ratio=1
        )


class ColabBlockOutputText(Paragraph):
    def __init__(self, text, **kwargs):
        super().__init__(text, font_size=COLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
                        line_spacing=0.5, **kwargs)

class ColabEnv(Mobject):
    def __init__(self, background=None):
        super().__init__()
        self.TOP_LEFT_CORNER = pixel2p(77, 156)
        self.env_image = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1) # background
        self.add(self.env_image)
        self.cells = []

    def add_cell(self, cell: ColabCodeBlock):
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=COLAB_SURROUND_CODE_BUFF)  # placeholdert buffer value
        self.cells.append(cell)
        self.add(cell)

    def remove_cell(self):
        if len(self.cells) > 0:
            self.remove(self.cells.pop())
            
    
    def clear(self):
        while len(self.cells) > 0:
            self.remove_cell()

    def outof_colab(self, cell: ColabCodeBlock, fullscreen=True, **kwargs):
        print('colab str:', cell.colabCode.code_string)
        target = ColabCode(cell.colabCode.code_string)
        if fullscreen:
            target.window.become(
                Rectangle(
                    fill_color=COLAB_LIGHTGRAY,
                    fill_opacity=1,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH))
        return AnimationGroup(
            Transform(cell.colabCode, target),
            FadeOut(self.env_image, cell.gutter, cell.playButton)
        )
    
    def focus_output(slef, cell: ColabCodeBlock, scale=0.75, fullscreen=True,**kwargs):
        if cell.outputWindow is None or cell.output is None:
            raise ValueError('Cell does not have output.')
        return AnimationGroup(
            cell.outputWindow.animate(**kwargs).become(
                Rectangle(
                color=WHITE,
                fill_opacity=1,
                height=FRAME_HEIGHT,
                width=FRAME_HEIGHT*4/3),
            ),
            cell.output.animate(**kwargs).scale_to_fit_width(FRAME_WIDTH*scale).center()
        ) 