'''Generic mooc utils'''

__all__ = [
    "FRAME_HEIGHT", "FRAME_WIDTH", "ASPECT_RATIO",
    "HALF_SCREEN_LEFT", "HALF_SCREEN_RIGHT",
    "SANS_SERIF_FONT", "CODE_FONT",
    "HighlightRectangle", "Title", "DynamicSplitScreen",
    "Cursor", "FunctionAbstraction", "VectorArray",
    "CustomDecimalNumber",
    "custom_get_axis_labels",
]

from manim import *
from manim.typing import Vector3D

FRAME_HEIGHT = 10.66  # In 4:3 frame height is 10.66, not 8!
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO
HALF_SCREEN_LEFT = [-FRAME_WIDTH/4, 0, 0]
HALF_SCREEN_RIGHT = [+FRAME_WIDTH/4, 0, 0]

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

class Title(Text):
    def __init__(self, text: str):
        super().__init__(
            text, color=BLACK,
            font_size=64, font=SANS_SERIF_FONT, weight=LIGHT)
        self.to_edge(UP).shift(UP*0.5)

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

    Since in many cases some parts of the content should not appear yet but the should be moved

    -if `direction` is UP, then the secondary rectangle weell move in/move out from the top.
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
                self.mainObj.animate(**kwargs).shift(shift)

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
        # return self.animate(rate_func=there_and_back, run_time=0.1).scale(0.8)
        return ApplyMethod(self.scale, 0.8, rate_func=there_and_back, run_time=0.1)
    
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
    def __init__(
        self,
        number: float = 0,
        font: str = None,
        **kwargs
    ):
        self.string_to_mob_map = {}  # presonal dict
        self.font = font
        super().__init__(number, **kwargs)

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

    def add_inputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF, relative_offset: float = 0.6):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs, relative_offset)
        self.InputArrows = VGroup(Arrow(ORIGIN, RIGHT*arrow_length, color=BLUE, stroke_width=6) for _ in range(n_inputs))
        self.InputArrows.arrange(DOWN, buff=spacing).next_to(self.Window, LEFT, buff=0)
        self.InputLabels = VGroup(
            Text(l, color=BLUE, font=CODE_FONT)
            if isinstance(l, str) else l  for l in labels
        )
        for i in range(n_inputs):
            self.InputLabels[i].next_to(self.InputArrows[i], LEFT, buff=buff)
        
        self.add(self.InputArrows, self.InputLabels)

    def add_outputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF, relative_offset: float = 0.6):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs, relative_offset)
        self.OutputArrows = VGroup(Arrow(ORIGIN, RIGHT*arrow_length, color=BLUE, stroke_width=6) for _ in range(n_inputs))
        self.OutputArrows.arrange(DOWN, buff=spacing).next_to(self.Window, RIGHT, buff=0)
        self.OutputLabels = VGroup(
            Text(l, color=BLUE, font=CODE_FONT)
            if isinstance(l, str) else l  for l in labels
        )
        for i in range(n_inputs):
            self.OutputLabels[i].next_to(self.OutputArrows[i], RIGHT, buff=buff)
        
        self.add(self.OutputArrows, self.OutputLabels)


class VectorArray(Table):
    def __init__(self, array, arrangement='vertical', include_dots=True, color=BLUE, h_buff=0.6, v_buff=1.0):
        table = [Text(t, font=CODE_FONT, color=BLACK) for t in array]
        if include_dots:
            if arrangement=='vertical':
                table.insert(-1, MathTex(r'\vdots', color=BLACK,stroke_width=4, stroke_color=BLACK))
            else:
                table.insert(-1, MathTex(r'\hdots', color=BLACK,stroke_width=4, stroke_color=BLACK))
        table = [[t] for t in table] if arrangement=='vertical' else [table]

        super().__init__(
            table, h_buff=h_buff, v_buff=v_buff,
            element_to_mobject= lambda m: m,  # identity
            include_outer_lines=False,
            line_config={'stroke_width':7, 'color':color}
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
