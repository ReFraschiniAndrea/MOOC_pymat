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
    "ColabCodeWithLogo",
    "draw_plot"
]

from manim import *
from .Generic_mooc_utils import FRAME_HEIGHT, FRAME_WIDTH, CODE_FONT, Cursor, FullScreenBackground
from .custom_code import CustomCode, CodeWithLogo
from typing import Any, List


# Colab colors for the editor
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

COLAB_FONT_SIZE = 12   # Font size for code in the editor

# Colab UI data
GOOGLE_FONT = "Google Sans Flex"  # Font of the Colab UI
_COLAB_UI_FONT_GRAY = "#1f1f1f"   # Color of the words in the Colab UI
_COLAB_FILE_ICON = r"Assets\colab_file_icon.png"
_COLAB_UI_FONT_SIZE = 12

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
        """Animate bringing a piece of code into a colab code cell.
        
        Parameters
        ----------
        colab_env : ColabEnv
            The environment object in which the code will appear as a code cell
        target_cell : int
            Index of the cell in the colab environment that the code will be placed into.
            The cell must already be added into the environment (and matching with the input).
            If not provided, a new matching cell is added to the envoronment
        
        """
        if target_cell is None:
            target = ColabCodeBlock(self.code_string)
            colab_env.add_cell(target, add_to_scene=False)
            cells_to_fade = colab_env.cells[:-1]
        else:
            target = colab_env.cells[target_cell]
            # cells_to_fade = colab_env.cells[:target_cell] + colab_env.cells[target_cell+1:]
            cells_to_fade = colab_env.cells[target_cell+1:]
        outputs_to_fade = Group(*[c.output for c in cells_to_fade if c.output is not None])
        if self.window is not None:
            return AnimationGroup(
                ReplacementTransform(self.window, target.window, **kwargs),
                ReplacementTransform(self.code, target.code, **kwargs),
                # if we fade in the whole environment, also the new cell will appear before it should
                FadeIn(colab_env.background, *cells_to_fade, outputs_to_fade, colab_env.cursor,
                       target.gutter, target.playButton, **kwargs)
            )
        else:
            return AnimationGroup(
                ReplacementTransform(self.code, target.code, **kwargs),
                FadeIn(colab_env.background, *cells_to_fade,outputs_to_fade, colab_env.cursor,
                       target.window, target.gutter, target.playButton, **kwargs)
            )


class ColabCodeBlock(ColabCode):
    def __init__(self, code: str):
        super().__init__(code, paragraph_config={'font_size': COLAB_FONT_SIZE})
        target_height = self.code.height + 2*_COLAB_SURROUND_CODE_BUFF if code != '' else 0.4766081925925926
        self.add_background_window(
            Rectangle(
                width=_COLAB_BLOCK_WIDTH,
                height= target_height,
                fill_color=COLAB_LIGHTGRAY, 
                fill_opacity=1,
                stroke_width=0
            )
        )
        self.gutter = self._create_Gutter().next_to(self.window, LEFT, buff=0)
        self.playButton = self._create_PlayButton().move_to(self.gutter.get_center()).align_to(self.gutter, UP).shift(DOWN*0.1)
        # be careful if the code is empty
        if self.code.family_members_with_points():
            self.code.align_to(self.window, LEFT).shift(RIGHT*_COLAB_GUTTER_TO_TEXT_BUFF)
        self.output = None
        self.outputWindow = None

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
            height=self.window.height,
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
        Group(self.outputWindow, self.output).next_to(VGroup(self.window, self.gutter), DOWN, buff=0).set_z_index(_COLAB_CELLS_Z_INDEX)
        self.output.align_to(self.code, LEFT)

        self.add(self.outputWindow)
        # self.add(self.output)

class ColabBlockOutputText(Paragraph):
    def __init__(self, text, **kwargs):
        super().__init__(text, font_size=COLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
                        line_spacing=0.5, **kwargs)

class ColabEnv():
    """Manager of the elements of the Colab UI."""
    def _pixel2p(x, y):
        '''Converts pixel coordinates (1080 x 1440) into manim units.'''
        return [
            (x- 720)/1440*FRAME_WIDTH,
            -(y - 540)/1080 *FRAME_HEIGHT,
            0
        ]
    
    PIXEL = FRAME_HEIGHT/1080
    TOP_LEFT_CORNER_ = _pixel2p(77, 156)
    # Button positions
    MENU_ = _pixel2p(25, 381)
    UPLOAD_ = _pixel2p(83, 206)
    PLUS_CODE_ = _pixel2p(210, 105)
    # Side menu COnstants
    SIDE_MENU_FIRST_FILE_ = _pixel2p(89, 319)
    SIDE_MENU_WIDTH_ = (425-50) * PIXEL
    FILE_ICON_HEIGHT_ = 26 * PIXEL
    FILE_TO_FILE_BUFFER_ = 16 * PIXEL
    ICON_TO_FILE_NAME_BUFFER_ = 10 * PIXEL


    def __init__(self, scene: Scene, background=None):
        self.scene : Scene = scene
        self.background : ImageMobject = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-4)
        self.cells : List[ColabCodeBlock] = []
        self.cursor : Cursor = Cursor().set_z_index(-2).move_to([-20,-20, 0])
        self.sidemenu : List[Mobject] = []

    def set_image(self, image_path: str):
        self.background.become(ImageMobject(image_path).scale_to_fit_height(FRAME_HEIGHT)).set_z_index(-4)

    def add_cell(self, cell: ColabCodeBlock = None, add_to_scene: bool = True):
        if cell is None:
            cell = ColabCodeBlock(code='')
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=_COLAB_SURROUND_CODE_BUFF)
        cell.set_z_index(_COLAB_CELLS_Z_INDEX)
        self.cells.append(cell)
        if add_to_scene:
            self.scene.add(cell)

    def get_cell(self, index: int) -> ColabCodeBlock:
        if index >= len(self.cells):
            raise IndexError(f'Cell index {index} out of range.')
        return self.cells[index]
    
    def get_cells(self) -> Group:
        return Group(*self.cells)

    def remove_cell(self):
        if len(self.cells) > 0:
            removed_cell = self.cells.pop()
            self._delete_cell(removed_cell)

    def _delete_cell(self, removed_cell: ColabCodeBlock):
        self.scene.remove(removed_cell)
        self.scene.remove(*removed_cell.submobjects)
        if removed_cell.output is not None:
            self.scene.remove(removed_cell.output)
            self.scene.remove(*removed_cell.output.submobjects)
        if removed_cell.outputWindow is not None:
            self.scene.remove(removed_cell.outputWindow)

    def remove_cell_from_top(self, n: int = 1):
        # Delete cells from the list beginning
        if len(self.cells) < n:
            raise ValueError("Not enough cells to be deleted")
        for _ in range(n):
            removed_cell = self.cells.pop(0)
            self._delete_cell(removed_cell)
        # Move the remaining cells to the top of the screen
        if len(self.cells) > 0:
            remaining_cells = self.get_cells()
            remaining_outputs = [c.output for c in remaining_cells if c.output is not None]
            remaining_cells.add(*remaining_outputs)
            remaining_cells.move_to(self.TOP_LEFT_CORNER_, UL)
    
    def clear(self):
        while len(self.cells) > 0:
            self.remove_cell()
        self.clear_sidemenu()
        self.clear_cursor()

    def clear_cursor(self):
        self.cursor.move_to([-20,-20, 0])
        self.scene.remove(self.cursor)

    def OutofColab(self, cell: ColabCodeBlock | int, fullscreen=True, **kwargs):
        if isinstance(cell, int):
            cell = self.get_cell(cell)

        target = ColabCode(cell.code_string)
        target.add_background_window()
        if fullscreen:
            target.window.become(
                Rectangle(
                    fill_color=COLAB_LIGHTGRAY,
                    fill_opacity=1,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH))
        cells_to_fade = self.cells[:-1]
        outputs_to_fade = Group(*[c.output for c in cells_to_fade if c.output is not None])
        return AnimationGroup(
            Transform(cell.code, target.code, **kwargs),
            Transform(cell.window, target.window, **kwargs),
            FadeOut(self.background, *cells_to_fade, outputs_to_fade, cell.gutter, cell.playButton, self.cursor, **kwargs)
        )
    
    def Run(
        self,
        cell: int = 0,
        new_cursor: bool = True
    ):
        cell_to_run = self.get_cell(cell)

        result = []
        if new_cursor:
            self.cursor.move_to(cell_to_run.playButton)
            result.append(GrowFromCenter(self.cursor))
        else:
            result.append(ApplyMethod(self.cursor.move_to, cell_to_run.playButton))
        
        result.append(self.cursor.Click())

        if cell_to_run.output is not None:
            # add the output so it appears above everything else
            # self.scene.add(cell_to_run.outputWindow, cell_to_run.output)
            result.extend([
                FadeIn(cell_to_run.outputWindow, cell_to_run.output, run_time=0),
                Wait(0.1)
            ])
 
        return Succession(*result)
    
    def FadeIn(self):
        return FadeIn(*self._get_obj_to_fade())
    def FadeOut(self):
        return FadeOut(*self._get_obj_to_fade())
    
    def _get_obj_to_fade(self):
        cells_to_fade = self.get_cells()
        outputs_to_fade = Group(*[c.output for c in cells_to_fade if c.output is not None])
        return [self.background, cells_to_fade, outputs_to_fade, self.cursor, *self.sidemenu]
    
    def FocusOutput(self, cell: int, scale=0.75, alignment=None, **kwargs):
        cell_to_focus = self.get_cell(cell)
        if cell_to_focus.output is None or cell_to_focus.outputWindow is None:
            raise ValueError("Selected cell has no output.")
        output = Group(cell_to_focus.outputWindow, cell_to_focus.output)
        output.set_z_index(0)  # any negative z_index will not work
        
        def _focus(output: Group):
            output[0].become(FullScreenBackground(WHITE))
            output[1].scale_to_fit_width(FRAME_WIDTH*scale).center()
            if alignment is not None:
                output[1].to_edge(alignment)
            return output
        return ApplyFunction(_focus, output, **kwargs)
    
    def add_file_to_sidemenu(self, name: str, add_to_scene: bool = True):
        file_icon = ImageMobject(_COLAB_FILE_ICON).scale_to_fit_height(self.FILE_ICON_HEIGHT_).set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])
        if len(self.sidemenu) == 0:
            file_icon.move_to(self.SIDE_MENU_FIRST_FILE_, aligned_edge=UL)
        else:
            file_icon.next_to(self.sidemenu[-1][0], DOWN, buff=self.FILE_TO_FILE_BUFFER_)
        file_name = Text(name, color=_COLAB_UI_FONT_GRAY, font=GOOGLE_FONT, font_size=_COLAB_UI_FONT_SIZE, weight=MEDIUM)
        file_name.next_to(file_icon, RIGHT, buff= self.ICON_TO_FILE_NAME_BUFFER_)
        file_name.shift(DOWN*3*self.PIXEL)

        self.sidemenu.append(Group(file_icon, file_name))
        if add_to_scene:
            self.scene.add(self.sidemenu[-1])
    

    def clear_sidemenu(self):
        for file in self.sidemenu:
            self.scene.remove(file)
            self.scene.remove(*file.submobjects)
        self.sidemenu = []

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

def draw_plot(fig) -> ImageMobject:
    """Convert matplotlib figure to ImageMobject"""
    fig.canvas.draw()
    buf1 = np.array(fig.canvas.buffer_rgba())
    return ImageMobject(buf1)