__all__ = [
    "MATLAB_GRAY",
    "MATLAB_LIGHTGRAY",
    "MATLAB_FONT_SIZE",
    "MatlabCode",
    "MatlabCodeBlock",
    "MatlabEnv",
    "MatlabOutputText",
    "MatlabCodeWithLogo",
]

from manim import *
from .Generic_mooc_utils import FRAME_WIDTH, FRAME_HEIGHT, CODE_FONT, Cursor, FullScreenBackground
from .custom_code import CustomCode, CodeWithLogo
from typing import Any, List

_MATLAB_LOGO = r'Assets\matlab_logo.png'
MATLAB_FONT_SIZE = 12
MATLAB_PLOT_WIDTH = 5
MATLAB_GRAY = "#f0f0f0"    # Color of plot windows
MATLAB_LIGHTGRAY = "#f7f7f7"
_MATLAB_CELL_TO_CELL_BUFF = 0.25
_MATLAB_CELLS_Z_INDEX = -3
_MATLAB_PLOT_Z_INDEX = -2

class MatlabCode(CustomCode):
    # override default options
    default_background_config: dict[str, Any] = {
        "buff": MED_SMALL_BUFF,
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

    def IntoMatlab(
            self,
            matlab_env: 'MatlabEnv',
            target_cell: int = None,
            **kwargs
        ):
        if target_cell is None:
            target = MatlabCodeBlock(self.code_string)
            matlab_env.add_cell(target)
            cells_to_fade = matlab_env.cells[:-1]
        else:
            target = matlab_env.get_cell(target_cell)
            cells_to_fade = matlab_env.cells[target_cell+1:]
        outputs_to_fade = matlab_env._get_shown_outputs()
        
        if self.window is not None:
            return AnimationGroup(
                ReplacementTransform(self.window, target.window, **kwargs),
                ReplacementTransform(self.code, target.code, **kwargs),
                FadeIn(matlab_env.background, *cells_to_fade, matlab_env.cursor, *outputs_to_fade, **kwargs),
        )
        else:
            return AnimationGroup(
                ReplacementTransform(self.code, target.code, **kwargs),
                FadeIn(matlab_env.background, *cells_to_fade, target.window, matlab_env.cursor, *outputs_to_fade, **kwargs)
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


class MatlabEnv():
    def _pixel2p(px, py):
        '''Converts pixel coordinates (1400 x 1050) into manim units.'''
        return [
             (px/1400 - 0.5)*FRAME_WIDTH,
            -(py/1050 - 0.5)*FRAME_HEIGHT,
            0
        ]
    
    TOP_LEFT_CORNER_ = _pixel2p(155, 241)
    TOP_LEFT_CORNER_UNSAVED_ = _pixel2p(128, 218)
    # OUTPUT_TOP_LEFT_CORNER_ = _pixel2p(80, 783)
    OUTPUT_TOP_LEFT_CORNER_ = _pixel2p(80, 833)

    NEW_SCRIPT_ = _pixel2p(25, 107)
    SAVE_ = _pixel2p(123, 71)
    SAVE_PROMPT_BUTTON_ = _pixel2p(874, 888)
    RUN_BUTTON_ = _pixel2p(1070, 67)
    BROWSE_FOLDER_ = _pixel2p(107, 170)
    OK_PROMPT_ = _pixel2p(874, 848)
    SIDEMENU_ = _pixel2p(23, 227)
    FIRST_SCRIPT_TAB_ = _pixel2p(105, 203)

    PIXEL = FRAME_WIDTH/1400
    OUTPUT_TO_OUTPUT_BUFF_ = 8*PIXEL

    def __init__(self, scene: Scene, background=None):

        self.scene = scene
        self.background : ImageMobject= ImageMobject(background).scale_to_fit_width(FRAME_WIDTH).center().set_z_index(-4)
        self.cells : List[MatlabCodeBlock] = []
        self.command_window_output : list[MatlabCommandWindowOutput] = []
        self.plot : MatlabPlot = None
        self.cursor : Cursor = Cursor().set_z_index(-2).move_to([-20,-20, 0])


    def set_image(self, image_path: str):
        self.background.become(ImageMobject(image_path).scale_to_fit_width(FRAME_WIDTH)).center().set_z_index(-4)

    def get_cell(self, index: int) -> MatlabCodeBlock:
        if index > len(self.cells):
            raise IndexError(f'Cell index {index} out of range.')
        return self.cells[index]
    
    def get_cells(self) -> Group:
        return Group(*self.cells)

    def add_cell(self, cell: MatlabCodeBlock = None, add_to_scene: bool = True):
        if cell is None:
            cell = MatlabCodeBlock(code='')
            if len(cell) > 0: cell.window.stretch_to_fit_height(25*self.PIXEL)
        if len(self.cells) == 0:
            cell.move_to(self.TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            cell.next_to(self.cells[-1], DOWN, buff=_MATLAB_CELL_TO_CELL_BUFF).align_to(self.cells[-1], LEFT)
        cell.set_z_index(_MATLAB_CELLS_Z_INDEX)
        self.cells.append(cell)
        if add_to_scene:
            self.scene.add(cell)

    def remove_cell(self):
        if len(self.cells) > 0:
            removed_cell = self.cells.pop()
            self._delete_cell(removed_cell)
            
    def _delete_cell(self, removed_cell: MatlabCodeBlock):
        self.scene.remove(removed_cell)
        self.scene.remove(*removed_cell.submobjects)

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
            remaining_cells.move_to(self.TOP_LEFT_CORNER_, UL)
    
    def remove_output(self):
        while len(self.command_window_output) > 0:
            removed_output = self.command_window_output.pop()
            self.scene.remove(removed_output)
            self.scene.remove(*removed_output.submobjects)
    
    def remove_plot(self):
        if self.plot is not None:
            self.scene.remove(self.plot)
            self.scene.remove(*self.plot.submobjects)
            self.plot = None
    
    def remove_cursor(self):
        self.cursor.move_to((-20,-20,0))
        self.scene.remove(self.cursor)
    
    def clear(self):
        while len(self.cells) > 0:
            self.remove_cell()
        self.remove_output()
        self.remove_plot()
        self.remove_cursor()

    def OutofMatlab(self, cell: MatlabCodeBlock | int, fullscreen=True, **kwargs):
        if isinstance(cell, int):
            cell = self.get_cell(cell)

        target = MatlabCode(cell.code_string)
        target.add_background_window()
        if fullscreen:
            target.window.become(
                Rectangle(
                    fill_color=WHITE,
                    fill_opacity=1,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH))
        cells_to_fade = self.cells[:-1]
        return AnimationGroup(
            Transform(cell, target),
            FadeOut(self.background, *cells_to_fade, *self._get_shown_outputs(), self.cursor),
    )

    def add_output_command_window(
        self,
        output_text: str | Mobject,
        add_to_scene: bool = False,
    ):
        if isinstance(output_text, str):
            output_text = MatlabOutputText(output_text)
        output = MatlabCommandWindowOutput(output_text)

        if len(self.command_window_output) == 0:
            output.move_to(self.OUTPUT_TOP_LEFT_CORNER_, aligned_edge=UL)
        else:
            output.next_to(self.command_window_output[-1], DOWN,buff=self.OUTPUT_TO_OUTPUT_BUFF_).align_to(self.command_window_output[-1], LEFT)
        
        output.set_z_index(_MATLAB_CELLS_Z_INDEX)
        self.command_window_output.append(output)
        if add_to_scene:
            self.scene.add(output)
            output._shown = True

    def add_output_plot(
        self,
        image: str | Mobject,
        image_width: float = MATLAB_PLOT_WIDTH,
        window_buff: float = 0.1,
        add_to_scene: bool = False
    ):
        if isinstance(image, str):
            image = ImageMobject(image)
            image.scale_to_fit_width(image_width).center()
        self.plot = MatlabPlot(image, buff=window_buff)
        self.plot.set_z_index(_MATLAB_PLOT_Z_INDEX)
        if add_to_scene:
            self.scene.add(self.plot)
        
    def add_output(
        self,
        output_text: str | Mobject = None,
        output_image: str | Mobject = None,
        add_to_scene: bool = False,
        **kwargs
    ):
        if output_text is not None:
            self.add_output_command_window(output_text, add_to_scene=add_to_scene)
        if output_image is not None:
            self.add_output_plot(output_image, add_to_scene=add_to_scene, **kwargs)

    def _get_shown_outputs(self):
        res = [output for output in self.command_window_output if output._shown]
        if self.plot is not None:
            res.append(self.plot)
        return res
        
    def Run(
        self,
        new_cursor: bool=True
    ):
        # Determine the outputs to show on run
        outputs_to_show = Group()
        for output in self.command_window_output:
            if not output._shown:
                outputs_to_show.add(output)
                output._shown = True
        if self.plot is not None:
            outputs_to_show.add(self.plot)

        if new_cursor:
            self.cursor.move_to(ORIGIN)
            return Succession(
                GrowFromCenter(self.cursor),
                ApplyMethod(self.cursor.move_to, self.RUN_BUTTON_),
                self.cursor.Click(),
                FadeIn(outputs_to_show, run_time=0),
                Wait(0.1)
            )
        else:
            return Succession(
                self.cursor.Click(),
                FadeIn(outputs_to_show, run_time=0),
                Wait(0.1)
            )

    def FocusOutput(
        self,
        output_to_focus: int = None,
        include_plot: bool = True,
        scale=0.75,
        buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER*2,
        alignment=None,
        **kwargs
    ):
        if len(self.command_window_output) == 0:
            raise ValueError("Matlab environment has no output")
        output_to_focus = self.command_window_output[-1] if output_to_focus is None else self.command_window_output[output_to_focus]

        if not include_plot or self.plot is None:
            def _focusOutput(m: MatlabCommandWindowOutput):
                m.object.scale_to_fit_width(FRAME_WIDTH*scale).center()
                m.outputWindow.become(FullScreenBackground(WHITE))
                return m
            return ApplyFunction(_focusOutput, output_to_focus)
        else:
            def _focusOutput(M: Group):
                M[0].object.scale_to_fit_width(FRAME_WIDTH*scale)
                M[1].next_to(M[0].object, DOWN, buff=buff)
                Group(M[0].object, M[1]).center()
                M[0].outputWindow.become(FullScreenBackground(WHITE))
                return M
            return ApplyFunction(_focusOutput, Group(output_to_focus, self.plot))
            

    def FocusPlot(self, scale=0.75, **kwargs):
        def _focusPlot(m: MatlabPlot):
            m.set_z_index(0)
            m.plotWindow.become(
                Rectangle(
                    color=WHITE,
                    height=FRAME_HEIGHT,
                    width=FRAME_WIDTH,
                    fill_opacity=1)
                )
            m.image.scale_to_fit_width(FRAME_WIDTH*scale).center()
            return m
        
        return ApplyFunction(_focusPlot, self.plot, **kwargs)
        
    
    def FadeOut(self):
        return FadeOut(self.background, self.get_cells(), *self._get_shown_outputs(), self.cursor)


class MatlabOutputText(Paragraph):
    def __init__(self, text, **kwargs):
        super().__init__(text, font_size=MATLAB_FONT_SIZE, color=BLACK, font=CODE_FONT,
                        line_spacing=0.5, **kwargs)

class MatlabCommandWindowOutput(Mobject):
    def __init__(self, object):
        super().__init__()
        self.object = object
        self.outputWindow = SurroundingRectangle(self.object, color=WHITE, corner_radius=0, fill_opacity=1, buff=0)
        self.add(self.outputWindow, self.object)
        self._shown : bool = False

class MatlabPlot(Mobject):
    def __init__(self, image: Mobject, buff: float = 0.1):
        super().__init__()
        self.image = image
        self.plotWindow = SurroundingRectangle(
            self.image, color=MATLAB_GRAY, 
            buff=buff, corner_radius=0.1,
            fill_opacity=1, stroke_width=0.5, stroke_color=BLACK
        )
        self.add(self.plotWindow, self.image)
        self.center()


class MatlabCodeWithLogo(CodeWithLogo):
    def __init__(
        self,
        code,
        logo_pos=UP,
        logo_buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        logo_shift_buff=0.5,
        **kwargs
    ):
        super().__init__(
            code_mobj=MatlabCode(code, **kwargs),
            logo_mobj=ImageMobject(_MATLAB_LOGO).scale(0.5),
            logo_pos=logo_pos,
            logo_buff=logo_buff,
            logo_shift_buff=logo_shift_buff
        )