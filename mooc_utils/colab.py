__all__ = [
    "COLAB_LIGHTGRAY",
    "COLAB_GRAY",
    "COLAB_DARKGRAY",
    "COLAB_GREEN",
    "COLAB_TEAL",
    "COLAB_PINE",
    "COLAB_PURPLE",
    "COLAB_BLUE",
    "COLAB_DEEPBLUE",
    "COLAB_BROWN",
    "COLAB_CRIMSON",
    "COLAB_FONT_SIZE",
    "ColabCode",
    "ColabCodeBlock",
    "ColabBlockOutputText",
    "ColabEnv",
    "ColabCodeWithLogo"
]

from manim import *
from .Generic_mooc_utils import FRAME_HEIGHT, FRAME_WIDTH, CODE_FONT, Cursor
from .custom_code import CustomCode, CodeWithLogo
from typing import Any, List


# Colab colors
COLAB_LIGHTGRAY = "#f7f7f7" # Main color of colab cell
COLAB_GRAY = "#eeeeee"  # Color of the left gutter of a colab cell
COLAB_DARKGRAY ="#424242" # color of the run button of a colab cell
COLAB_GREEN = "#007900"
COLAB_TEAL = "#257693"
COLAB_PINE = "#116644"
COLAB_PURPLE = "#cf70e7"
COLAB_BLUE = "#0431fa"
COLAB_DEEPBLUE = "#001080"
COLAB_BROWN = "#795e26"
COLAB_CRIMSON = "#a31515"

COLAB_FONT_SIZE = 12

# colab environment constants
_COLAB_LEFT_BUFF = 127/1440 * FRAME_WIDTH  # Distance of a colab cell from the left edge of the screen
_COLAB_GUTTER_WIDTH = 50/1080 *FRAME_HEIGHT # Width of the left gutter
_COLAB_BLOCK_WIDTH = 1300/1440 * FRAME_WIDTH
_COLAB_BUTTON_RADIUS = (23/1080*FRAME_HEIGHT)/2
_COLAB_SURROUND_CODE_BUFF = 17/1080*FRAME_HEIGHT
_COLAB_GUTTER_TO_TEXT_BUFF = 10/1440 * FRAME_WIDTH
_PYTHON_LOGO = r'Assets\python_logo.png'
_COLAB_CELLS_Z_INDEX = -3

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
            language='colabpython',
            formatter_style='colab',
            paragraph_config=paragraph_config
        )

    def IntoColab(
        self,
        colab_env: 'ColabEnv',
        target_cell: int = None,
        **kwargs
    ):
        if target_cell is None:
            target = ColabCodeBlock(self.code_string)
            colab_env.add_cell(target)
            cells_to_fade = colab_env.cells[:-1]
        else:
            target = colab_env.cells[target_cell]
            # cells_to_fade = colab_env.cells[:target_cell] + colab_env.cells[target_cell+1:]
            cells_to_fade = colab_env.cells[target_cell+1:]
        if self.window is not None:
            return AnimationGroup(
                ReplacementTransform(self.window, target.colabCode.window, **kwargs),
                ReplacementTransform(self.code, target.colabCode.code, **kwargs),
                # if we fade in the whole environment, also the new cell will appear before it should
                FadeIn(colab_env.env_image, *cells_to_fade, colab_env.cursor,
                       target.gutter, target.playButton, **kwargs)
            )
        else:
            return AnimationGroup(
                ReplacementTransform(self.code, target.colabCode.code, **kwargs),
                FadeIn(colab_env.env_image, *cells_to_fade, colab_env.cursor,
                       target.colabCode.window, target.gutter, target.playButton, **kwargs)
            )


class ColabCodeBlock(Mobject):
    def __init__(self, code: str):
        super().__init__()
        self.colabCode = ColabCode(code, paragraph_config={'font_size': COLAB_FONT_SIZE})
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

    def _create_PlayButton(self):
        button = Circle(
            color=COLAB_DARKGRAY, 
            radius=_COLAB_BUTTON_RADIUS,
            fill_opacity=1,
            stroke_width=0)
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
        Group(self.outputWindow, self.output).next_to(VGroup(self.colabCode.window, self.gutter), DOWN, buff=0).set_z_index(_COLAB_CELLS_Z_INDEX)
        self.output.align_to(self.colabCode.code, LEFT)

        self.add(self.outputWindow)
        self.add(self.output)
    
    def focus(self, scale=0.75, alignment=None):
        '''Can be used with animate keyword'''
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
        if alignment is not None:
            self.output.to_edge(alignment)

class ColabBlockOutputText(Paragraph):
    def __init__(self, text, **kwargs):
        super().__init__(text, font_size=COLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
                        line_spacing=0.5, **kwargs)

class ColabEnv(Mobject):
    def _pixel2p(x, y):
        '''Converts pixel coordinates (1080 x 1440) into manim units.'''
        return [
            (x- 720)/1440*FRAME_WIDTH,
            -(y - 540)/1080 *FRAME_HEIGHT,
            0
        ]
    
    TOP_LEFT_CORNER_ = _pixel2p(77, 156)
    MENU_ = _pixel2p(25, 381)
    UPLOAD_ = _pixel2p(83, 206)
    PLUS_CODE_ = _pixel2p(210, 105)

    def __init__(self, background=None):
        super().__init__()
        self.env_image = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-4)
        self.add(self.env_image)
        self.cells : List[ColabCodeBlock] = []
        self.cursor = Cursor().set_z_index(-2).move_to([-20,-20, 0])
        self.add(self.cursor)

    def set_image(self, image_path: str):
        self.env_image.become(ImageMobject(image_path).scale_to_fit_height(FRAME_HEIGHT)).set_z_index(-4)

    def add_cell(self, cell: ColabCodeBlock):
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=_COLAB_SURROUND_CODE_BUFF)
        cell.set_z_index(_COLAB_CELLS_Z_INDEX)
        self.cells.append(cell)
        self.add(cell)

    def remove_cell(self, scene: Scene):
        if len(self.cells) > 0:
            removed_cell = self.cells.pop()
            self.remove(removed_cell)  
            scene.remove(removed_cell)
    
    def clear(self, scene: Scene):
        while len(self.cells) > 0:
            self.remove_cell(scene)
        # cursor is removed from the scene but not from the environment
        self.cursor.move_to([-20,-20, 0])
        scene.remove(self.cursor)

    def OutofColab(self, cell: ColabCodeBlock, fullscreen=True, **kwargs):
        target = ColabCode(cell.colabCode.code_string)
        target.add_background_window()
        if fullscreen:
            target.window.become(
                Rectangle(
                    fill_color=COLAB_LIGHTGRAY,
                    fill_opacity=1,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH))
        cells_to_fade = self.cells[:-1]
        return AnimationGroup(
            Transform(cell.colabCode, target),
            FadeOut(self.env_image, *cells_to_fade,cell.gutter, cell.playButton, self.cursor)
        )
    
    def Run(
        self,
        cell: int = 0,
        new_cursor: bool = True
    ):
        if cell > len(self.cells):
            raise IndexError('Cell index out of range')
        cell_to_run = self.cells[cell]

        result = []
        if new_cursor:
            self.cursor.move_to(cell_to_run.playButton)
            result.append(GrowFromCenter(self.cursor))
        else:
            result.append(ApplyMethod(self.cursor.move_to, cell_to_run.playButton))
        
        result.append(self.cursor.Click())

        if cell_to_run.output is not None:
            # add again the output so it is above everything else
            cell_to_run.add(cell_to_run.outputWindow, cell_to_run.output)
            result.extend([
                FadeIn(cell_to_run.outputWindow, cell_to_run.output, run_time=0),
                Wait(0.1)
            ])
 
        return Succession(*result)
    
    def focus_output(self, cell: int, scale=0.75, alignment=None, **kwargs):
        if cell > len(self.cells):
            raise IndexError("Cell index out of range.")
        if self.cells[cell].output is None:
            raise ValueError("Selected cell has no output")
        self.cells[cell].outputWindow.set_z_index(0)
        self.cells[cell].output.set_z_index(0)  # any negative z_index will not work
        return self.cells[cell].animate(**kwargs).focus(scale, alignment)

class ColabCodeWithLogo(CodeWithLogo):
    def __init__(
        self,
        code,
        logo_pos=UP,
        logo_buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        logo_shift_buff = 0.5,
        **kwargs
    ):
        super().__init__(
            code_mobj=ColabCode(code, **kwargs),
            logo_mobj=ImageMobject(_PYTHON_LOGO).scale(0.5),
            logo_pos=logo_pos,
            logo_buff=logo_buff,
            logo_shift_buff=logo_shift_buff
        )