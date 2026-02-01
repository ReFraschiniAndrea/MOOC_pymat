'''Generic mooc utils'''

__all__ = [
    "FRAME_HEIGHT", "FRAME_WIDTH", "ASPECT_RATIO",
    "HALF_SCREEN_LEFT", "HALF_SCREEN_RIGHT", "TITLED_CENTER",
    "SANS_SERIF_FONT", "CODE_FONT",
    "HighlightRectangle", "FullScreenBackground","SlideTitle", "DynamicSplitScreen",
    "Cursor", "FunctionAbstraction",
    "VectorArray", "PixelArray",
    "CustomDecimalNumber",
    "custom_get_axis_labels",
]

from manim import *
from manim.typing import Vector3D
import itertools as it

FRAME_HEIGHT = 8*4/3 # In 4:3 frame height is 10.666.., not 8! 
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO  # The frame width is the same: 8*16/9 = 8*(4/3)*(4/3)
HALF_SCREEN_LEFT = np.array([-FRAME_WIDTH/4, 0, 0])
HALF_SCREEN_RIGHT = np.array([+FRAME_WIDTH/4, 0, 0])

TITLE_DOWN_ALIGNMENT = 3.75
TITLED_CENTER = DOWN * (FRAME_HEIGHT/4 - TITLE_DOWN_ALIGNMENT/2)
SANS_SERIF_FONT = 'Arial'
CODE_FONT = 'Aptos Mono'

_CURSOR_ICON = r'Assets\classic_cursor.svg'
_LAPTOP_ICON = r"Assets\laptop_icon.svg"


def custom_get_axis_labels(
    ax: Axes,
    x_label: Mobject,
    y_label: Mobject
):
    return VGroup(
        x_label.next_to(ax.get_axis(0).get_corner(UR), UP),
        y_label.next_to(ax.get_axis(1).get_corner(UR), RIGHT),
    )

class SlideTitle(Text):
    def __init__(self, text: str):
        super().__init__(
            text, color=BLACK,
            font_size=64, font=SANS_SERIF_FONT, weight=LIGHT)
        self.center()
        self.shift(UP*(TITLE_DOWN_ALIGNMENT - self[0].get_bottom()[1]))

class HighlightRectangle(BackgroundRectangle):
    def __init__(
        self,
        mobject: Mobject,
        color = BLUE,
        opacity: float = 0.4,
        corner_radius: float = 0.1,
        buff: float = 0.05,
        **kwargs
    ):
        super().__init__(mobject, color=color, 
                         stroke_width=0, stroke_opacity=0, fill_opacity=opacity, 
                         buff=buff, corner_radius=corner_radius, **kwargs)
        self.set_z_index(mobject.z_index)
        mobject.set_z_index(mobject.z_index+0.1)

class FullScreenBackground(Rectangle):
    def __init__(self, color=WHITE, **kwargs):
        super().__init__(
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            color=color, fill_opacity=1, stroke_width=0, **kwargs)
        self.set_z_index(-1)
        
class DynamicSplitScreen(Mobject):
    """Horizontal spliscreen that adapts dynamically to the content.
    
    The DSS is formed by two horizontal rectnagle, a mainRect and a secondaryRect
    - The mainRect covers the entire screen and is intended to hold the code that
    is written during the lecture
    - The secondaryRect holds instead supporting material (e.g. formulas) needed to understand the code.
    
    The main animation of this object is bringIn/bringOut: BringIn slides the secondaryRect with its
    material into frame, and moves the content of the mainRect accordingly so that it is still centered;
    bringOut does the opposite.

    To manage what objects are affected by the DSS, there are 2 methods: add_main_obj,
    add_side_obj and the respective remove methods.

    Since in many cases some parts of the content should not appear yet but they should be moved
    -follow_obj: object that should be moved exaclty like the main object, but it has not
    appeared yet in the scene, and so it should not be animated while moved
    -condsider_follow: sometimes, the follow_obj appears much later in the scene, and we
    want to still move it entirely but consider only some part of it for the calculations
    (centering on main rectangle). consider_follow is the part of the follow_obj actually
    taken into account for calculations. (By default, the entire follow_obj)

    -if `direction` is UP, then the secondary rectangle will move in/move out from the top.
    if it is instead DOWN, it does so from the bottom of the screen.
    """
    def __init__(
        self,
        main_color = BLUE,
        side_color = RED,
        buff = SMALL_BUFF*2,
        direction : Vector3D = UP
    ):
        super().__init__()
        self.DIRECTION_ = direction

        self.mainRect = Rectangle(
            color=main_color, 
            width=FRAME_WIDTH, 
            height=FRAME_HEIGHT, 
            fill_opacity=1, 
            stroke_width=0
        ).set_z_index(-1).center()
        self.secondaryRect = Rectangle(
            color=side_color, 
            width=FRAME_WIDTH,
            height= 2 * buff,
            fill_opacity=1,
            stroke_width=0
        ).set_z_index(-1).move_to(self.mainRect.get_edge_center(self.DIRECTION_), aligned_edge = -self.DIRECTION_)
        self.mainRect.save_state()
        self.secondaryRect.save_state()

        self.mainObj = None
        self.followMainObj = None
        self.secondaryObj = None
        self.followSecondaryObj = None
        self.brought_in_ = False
        self.last_shift_ = None
        self.buff_ = buff

        self.mainRect.add_updater(
            lambda r: r.stretch_to_fit_height(
                self.secondaryRect.get_edge_center(-self.DIRECTION_)[1] * self.DIRECTION_[1]
                + FRAME_HEIGHT/2 + 1/1080*FRAME_HEIGHT
                ).move_to(-FRAME_HEIGHT/2 *self.DIRECTION_, aligned_edge=-self.DIRECTION_)
        )
        self.add(self.mainRect, self.secondaryRect)

    def add_main_obj(self, main_obj: VMobject, follow_obj: VMobject = None):
        self.mainObj = main_obj
        self.followMainObj = follow_obj 

    def remove_main_obj(self):
        self.mainObj = None
        self.followMainObj = None

    def add_side_obj(
        self, secondary_object: VMobject,
        follow_side_obj: Mobject = None,
        consider_follow: Mobject = None,
        center_horizontally: bool = True):
        """If the secondary rectangle is out of frame, resizes it and adds the object
        If it's in frame, the rectangle is not resized and it is assumed that the the
        object is already in the correct position
        """
        self.remove_side_obj()

        if self.brought_in_:
            self.secondaryRect.move_to(self.DIRECTION_ *FRAME_HEIGHT/2, aligned_edge=self.DIRECTION_)
            self.mainRect.update()
        else:
            # compute the new height of the secondary rectangle
            if consider_follow is not None:
                _sideObj = Group(secondary_object, consider_follow)
            elif follow_side_obj is not None:
                _sideObj = Group(secondary_object, follow_side_obj)
            else:
                _sideObj = secondary_object

            self.secondaryRect.stretch_to_fit_height(_sideObj.height + 2 * self.buff_)
            self.secondaryRect.move_to(self.mainRect.get_edge_center(self.DIRECTION_), aligned_edge = -self.DIRECTION_)

            # compute shift to move the side object(s) into position
            if center_horizontally:
                _shift = self.secondaryRect.get_center() - _sideObj.get_center()
            else:
                _shift = (self.secondaryRect.get_y() - _sideObj.get_y())*UP
            
            # NOTE: might want to move explicitly move follow object too
            _toMove = Group(secondary_object)
            if consider_follow is not None: _toMove.add(consider_follow)
            if follow_side_obj is not None: _toMove.add(follow_side_obj)
            _toMove.shift(_shift)

        self.secondaryObj = secondary_object
        self.followSecondaryObj = follow_side_obj
        self.add(self.secondaryObj)

    def add_empty_side_obj(self, height):
        """Height is intended to be the one of the objects that will appear."""
        self.remove_side_obj()
        self.secondaryRect.stretch_to_fit_height(height)
        if self.brought_in_ == False:
            self.secondaryRect.move_to(self.mainRect.get_edge_center(self.DIRECTION_), aligned_edge = -self.DIRECTION_)
        else:
            self.secondaryRect.move_to(self.DIRECTION_ *FRAME_HEIGHT/2, aligned_edge=self.DIRECTION_)
            self.mainRect.update()
    
    def remove_side_obj(self):
        if self.secondaryObj is not None:
            self.remove(self.secondaryObj)
            self.secondaryObj = None
        self.followSecondaryObj = None

    def reset(self):
        self.remove_main_obj()
        self.remove_side_obj()
        self.brought_in_=False
        self.secondaryRect.restore()
        self.mainRect.restore()
        self.mainRect.resume_updating()
        self.last_shift_ = None

    def _get_shift(self, consider_follow: Mobject = None):
        if self.brought_in_: 
            final_pos = -self.secondaryRect.height/2 * self.DIRECTION_
        else:
            final_pos = ORIGIN

        full_obj = Group()
        if self.mainObj is not None:
            full_obj.add(self.mainObj)
        if consider_follow is not None:
            full_obj.add(consider_follow)
        elif self.followMainObj is not None:
            # by default, consider the entire followobject
            full_obj.add(self.followMainObj)

        shift = final_pos[1] - full_obj.get_y()
        return shift*UP
    
    def _MainObjIntoPosition(self, consider_follow: Mobject = None, animate: bool = True, **kwargs):
        if self.mainObj is None and self.followMainObj is None:
            return None
        shift = self._get_shift(consider_follow)
        
        if self.followMainObj is not None:
            self.followMainObj.shift(shift)
        
        if self.mainObj is not None:
            if animate:
                return self.mainObj.animate(**kwargs).shift(shift)
            else:
                self.mainObj.shift(shift)

        return None
    
    def _MoveSecondaryRect(self, direction, animate: bool = True, **kwargs):
        secondary_group = Group(self.secondaryRect)
        if self.secondaryObj is not None:
            secondary_group.add(self.secondaryObj)

        if self.followSecondaryObj is not None:
            self.followSecondaryObj.shift(direction*self.secondaryRect.height)

        if animate:
            return secondary_group.animate(**kwargs).shift(direction*self.secondaryRect.height)
        else:
            secondary_group.shift(direction*self.secondaryRect.height)
            return None
    
    def bringIn(self, consider_follow: Mobject = None, **kwargs):
        """Bring secondary rectangle into frame"""
        self.brought_in_=True
        return self._Bring(-self.DIRECTION_, consider_follow,  **kwargs)
    
    def bringOut(self, consider_follow: Mobject = None, **kwargs):
        """Bring secondary rectangle out of frame"""
        self.brought_in_=False
        return self._Bring(self.DIRECTION_, consider_follow,  **kwargs)

    def _Bring(self, direction, consider_follow: Mobject = None, animate: bool = True, **kwargs):
        secondary_shift = self._MoveSecondaryRect(direction=direction, animate=animate, **kwargs)
        main_shift = self._MainObjIntoPosition(consider_follow, animate=animate, **kwargs)
        if main_shift is None:
            return secondary_shift
        else: 
            return AnimationGroup(secondary_shift, main_shift)

    def hard_bring_in(self, consider_follow: Mobject=None):
        if not self.brought_in_:
            self.brought_in_=True
            self._Bring(-self.DIRECTION_, consider_follow, animate=False)

    def hard_bring_out(self, consider_follow: Mobject=None):
        if self.brought_in_:
            self.brought_in_=False
            self._Bring(self.DIRECTION_, consider_follow, animate=False)

class Cursor(SVGMobject):
    """Classic hand cursor for tutorial animations.
    
    Has a 'click' method animation anda modified 'move_to' so that motion is w.r.t the fingertip.
    """
    def __init__(self, **kwargs):
        super().__init__(file_name=_CURSOR_ICON, height=(24/1080)*FRAME_HEIGHT, **kwargs)
        self.stroke_width = 0  # avoid some bugs

    def Click(self):
        return ApplyMethod(self.scale, 0.8, rate_func=there_and_back, run_time=0.1)
    
    def MoveAndClick(self, point,**kwargs):
        return Succession(
            ApplyMethod(self.move_to, point, **kwargs),
            self.Click()
        )
    
    def fingertip(self):
        return self.get_top() + LEFT * 2.5/17*self.width# + DOWN*100/1200*self.height+
    
    def move_to(self, point_or_mobject):
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_critical_point(ORIGIN)
        else:
            target = point_or_mobject
        self.shift(target - self.fingertip())
        return self


class CustomDecimalNumber(DecimalNumber):
    '''Override of default decimal number to allow for different fonts.'''
    _FONT_STRING_TO_MOB_MAPS = {}  # class member
    def __init__(
        self,
        number: float = 0,
        font: str = '',
        mob_class = Text,
        **kwargs
    ):
        self.font = font
        if self.font not in CustomDecimalNumber._FONT_STRING_TO_MOB_MAPS.keys():
            CustomDecimalNumber._FONT_STRING_TO_MOB_MAPS[font] = {}
        self.string_to_mob_map = CustomDecimalNumber._FONT_STRING_TO_MOB_MAPS[font]
        super().__init__(number, **kwargs, mob_class=mob_class)
        self._set_submobjects_from_number(number) # update with new font
        self.set_color(self.color)

    def _string_to_mob(self, string: str, mob_class: VMobject | None = None, **kwargs):
        if mob_class is None:
            mob_class = self.mob_class

        if string not in self.string_to_mob_map:
            if self.mob_class == Text and self.font is not None:
                self.string_to_mob_map[string] = mob_class(string, font = self.font, **kwargs)
            else:
                self.string_to_mob_map[string] = mob_class(string, **kwargs)
        mob = self.string_to_mob_map[string].copy()
        mob.font_size = self._font_size
        return mob

class FunctionAbstraction(VMobject):
    def __init__(self, scale = 1):
        super().__init__()
        self.LaptopIcon = SVGMobject(_LAPTOP_ICON).set_color(BLUE).scale(scale)
        self.Window = SurroundingRectangle(
            self.LaptopIcon,
            fill_color=WHITE,
            stroke_color=BLUE,
            buff=0.5,
            stroke_width = 6,
            fill_opacity=1
        )
        self.add(self.Window, self.LaptopIcon)

    def _get_spacing(self, n: int, relative_offset: float = 0.6):
        return relative_offset*self.Window.height / (n-1) if n > 1 else 0

    def add_inputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF, relative_offset: float = 0.6, **kwargs):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs, relative_offset)
        self.InputArrows = VGroup(Arrow(ORIGIN, RIGHT*arrow_length, color=BLUE, stroke_width=6) for _ in range(n_inputs))
        self.InputArrows.arrange(DOWN, buff=spacing).next_to(self.Window, LEFT, buff=0)
        self.InputLabels = VGroup(
            Text(l, color=BLUE, font=CODE_FONT, **kwargs)
            if isinstance(l, str) else l  for l in labels
        )
        for i in range(n_inputs):
            self.InputLabels[i].next_to(self.InputArrows[i], LEFT, buff=buff)
        
        self.add(self.InputArrows, self.InputLabels)

    def add_outputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF, relative_offset: float = 0.6, **kwargs):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs, relative_offset)
        self.OutputArrows = VGroup(Arrow(ORIGIN, RIGHT*arrow_length, color=BLUE, stroke_width=6) for _ in range(n_inputs))
        self.OutputArrows.arrange(DOWN, buff=spacing).next_to(self.Window, RIGHT, buff=0)
        self.OutputLabels = VGroup(
            Text(l, color=BLUE, font=CODE_FONT, **kwargs)
            if isinstance(l, str) else l  for l in labels
        )
        for i in range(n_inputs):
            self.OutputLabels[i].next_to(self.OutputArrows[i], RIGHT, buff=buff)
        
        self.add(self.OutputArrows, self.OutputLabels)


class VectorArray(Table):
    def __init__(
            self,
            array,
            arrangement='vertical',
            elem_to_mob_class=Text,
            elem_to_mob_config: dict = {'font':CODE_FONT, 'color':BLACK},
            include_dots=True,
            color=BLUE,
            line_config={},
            h_buff=0.6, v_buff=1.0):
        table = [elem_to_mob_class(t ,**elem_to_mob_config) for t in array]
        if include_dots:
            if arrangement=='vertical':
                table.insert(-1, MathTex(r'\vdots', color=BLACK,stroke_width=4, stroke_color=BLACK))
            else:
                table.insert(-1, MathTex(r'\hdots', color=BLACK,stroke_width=4, stroke_color=BLACK))
        table = [[t] for t in table] if arrangement=='vertical' else [table]

        line_kwargs = {'stroke_width':7, 'color':color}
        line_kwargs.update(line_config)
        super().__init__(
            table, h_buff=h_buff, v_buff=v_buff,
            element_to_mobject= lambda m: m,  # identity
            include_outer_lines=False,
            line_config=line_kwargs
        )
        # The outer rectangle should be added first so it is drawn first
        _lines =  self.get_horizontal_lines() + self.get_vertical_lines()
        _entries = self.get_entries()
        self.remove(_lines, _entries)
        self._add_outer_rectangle()
        self.add(_lines, _entries)
        
    def _add_outer_rectangle(self):
        anchor_left = self.get_columns()[0].get_left()[0] - 0.5 * self.h_buff
        anchor_right = self.get_columns()[-1].get_right()[0] + 0.5 * self.h_buff
        anchor_top = self.get_rows()[0].get_top()[1] + 0.5 * self.v_buff
        anchor_bottom = self.get_rows()[-1].get_bottom()[1] - 0.5 * self.v_buff
        self.outer_rectangle = Polygon(
            [anchor_left, anchor_top, 0],
            [anchor_right, anchor_top, 0],
            [anchor_right, anchor_bottom, 0],
            [anchor_left, anchor_bottom, 0],
            **self.line_config
        )
        self.add(self.outer_rectangle)

    def get_lines(self) -> VGroup:
        return VGroup(self.outer_rectangle) + self.get_horizontal_lines() + self.get_vertical_lines()


class TabularVGroup(VGroup):
    def __init__(self, *objects, height, width):
        super().__init__(objects)
        self.tab_width : int=int(width)
        self.tab_height : int=int(height)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.submobjects[key[0]*self.tab_width+key[1]]
        return super().__getitem__(key)

    def get_row(self, i: int):
        if i < 0 or i > self.tab_height:
            return
        row = VGroup(self[i,j] for j in range(self.tab_width))
        return row
    
    def get_column(self, j: int):
        if j < 0 or j > self.tab_width:
            return
        column = VGroup(self[i,j] for i in range(self.tab_height))
        return column

class PixelArray(VGroup):
    """Image-Table-Matrix hybrid object"""
    def __init__(
        self,
        array : np.ndarray,
        **kwargs
    ):
        super().__init__()
        self.array : np.ndarray = array  # Grayscale or RGB values
        self.pixel_array : TabularVGroup = None  # squares for the pixels
        self.pixel_values : TabularVGroup = None
        self.brackets : VGroup = None

        self.add_pixel_array(self.array, **kwargs)

    def add_pixel_array(
        self,
        array,
        stroke_width: float = 1,
        stroke_color = None,
        height: float = None
    ):
        h, w = array.shape[:2]
        # Create a grid of square pixels
        pixel_array = TabularVGroup(*[Square() for _ in range(h*w)], width=w, height=h)
        if array.ndim == 2:
            for pixel, value in zip(pixel_array, array.flatten()):
                color = rgb_to_color((value, value, value))
                pixel.set_fill(color, 1.0)
                pixel.set_stroke(color=color)
        else:
            for pixel, value in zip(pixel_array, it.chain(*array)):
                color = rgb_to_color(value)
                pixel.set_fill(color, 1.0)
                pixel.set_stroke(color=color)

        pixel_array.set_stroke(color=stroke_color, width=stroke_width)
        pixel_array.arrange_in_grid(h, w, buff=0)
        if height is not None:
            pixel_array.set_height(height)
        
        self.pixel_array = pixel_array
        self.add(self.pixel_array)

    def get_pixel_highlight(self, color=YELLOW, stroke_width=2, position = (0,0)) -> Square:
        """Get square border to highlight a single pixel"""
        pixel_highlight : Square = self.pixel_array[*position].copy()
        pixel_highlight.set_fill(BLACK, 0)
        pixel_highlight.set_stroke(color, stroke_width, opacity=1)
        pixel_highlight.set_z_index(self.pixel_array.z_index + 1)
        return pixel_highlight

    def get_kernel_array(
        self,
        kernel: np.ndarray,
        kernel_color=YELLOW,
        kernel_stroke_width=2,
        add_values:bool = True,
        element_to_mob_class =  DecimalNumber,
        values_size_fator : float =0.3,
        kernel_tex = None,
        **kwargs
    ) -> TabularVGroup:
        """Get highlight in the shape of the convolution kernel"""
        pixel_highlight = self.get_pixel_highlight(kernel_color, kernel_stroke_width)
        kernel_array = TabularVGroup(height=kernel.shape[0], width=kernel.shape[1])
        for _ in kernel.flatten():
            kernel_array.add(pixel_highlight.copy())
        kernel_array.arrange_in_grid(*kernel.shape, buff=0)
        kernel_array.move_to(self.pixel_array[0, 0])
       
        if not add_values:
            return kernel_array
        
        values = TabularVGroup(height=kernel.shape[0], width=kernel.shape[1])
        for i, x in enumerate(kernel.flatten()):
            if kernel_tex:
                value = MathTex(kernel_tex, color=kernel_color, **kwargs)
            else:
                value = element_to_mob_class(x, **kwargs)
            value.scale_to_fit_height(pixel_highlight.height * values_size_fator)
            value.move_to(kernel_array[i])
            values.add(value)

        return kernel_array, values
 
    def add_pixel_values(
        self,
        num_decimal_places=0,
        color=RED
    ): 
        values = TabularVGroup(height=self.array.shape[0], width=self.array.shape[1])
        for pixel, value in zip(self.pixel_array, self.array.flatten()):
            pixel_value = CustomDecimalNumber(value,font=CODE_FONT, mob_class = Text, num_decimal_places=num_decimal_places, color=color)
            pixel_value.set_height(pixel.height*0.3)
            pixel_value.move_to(pixel)
            values.add(pixel_value)

        self.pixel_values = values
        self.add(self.pixel_values)
    
    def add_brackets(
        self,
        left: str = "[",
        right: str = "]",
        bracket_h_buff: float = MED_SMALL_BUFF,
        bracket_v_buff: float = MED_SMALL_BUFF,
        **kwargs
    ):
        # Height per row of LaTeX array with default settings
        BRACKET_HEIGHT = 0.5977
        n = int((self.pixel_values.height) / BRACKET_HEIGHT) + 1
        empty_tex_array = "".join(
            [ r"\begin{array}{c}", *n * [r"\quad \\"], r"\end{array}", ]
        )
        tex_left = "".join(
            [ r"\left" + left, empty_tex_array, r"\right.", ]
        )
        tex_right = "".join(
            [ r"\left.", empty_tex_array, r"\right" + right, ]
        )
        l_bracket = MathTex(tex_left, **kwargs)
        r_bracket = MathTex(tex_right, **kwargs)

        self.brackets = VGroup(l_bracket, r_bracket)
        # if self.stretch_brackets:
        self.brackets.stretch_to_fit_height(self.pixel_values.height + 2 * bracket_v_buff)
        l_bracket.next_to(self.pixel_values, LEFT, bracket_h_buff)
        r_bracket.next_to(self.pixel_values, RIGHT,bracket_h_buff)

        self.add(self.brackets)

    def remove_brackets(self):
        self.remove(self.brackets)
        self.brackets = None

    def add_outline(self, color=DARK_BLUE, stroke_width=4):
        self.outline = SurroundingRectangle(self.pixel_array, buff=0, fill_opacity=0, color=color, stroke_width=stroke_width)
        self.set_z_index(1)
        self.outline.move_to(self[0]).set_z_index(0)
        self.add_to_back(self.outline)