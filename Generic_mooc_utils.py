'''Generic mooc utils'''

__all__ = [
    "FRAME_HEIGHT", "FRAME_WIDTH", "ASPECT_RATIO",
    "HALF_SCREEN_LEFT", "HALF_SCREEN_RIGHT",
    "SANS_SERIF_FONT", "CODE_FONT",
    "HighlightRectangle", "Title", "DynamicSplitScreen",
    "Cursor", "FunctionAbstraction", "VectorArray",
    "CustomDecimalNumber",
    "custom_get_axis_labels", "pixel2p",
]

from manim import *
import manimpango

FRAME_HEIGHT = 10.66  # In 4:3 frame height is 10.66, not 8!
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO
HALF_SCREEN_LEFT = [-FRAME_WIDTH/4, 0, 0]
HALF_SCREEN_RIGHT = [+FRAME_WIDTH/4, 0, 0]

SANS_SERIF_FONT = 'Arial'
CODE_FONT = 'Aptos Mono'
try:
    manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos-Mono.ttf")
    manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos.ttf")
except:
    print('warning, unable to find font. falling back to monospace.')
    CODE_FONT = 'Monospace'

_CURSOR_ICON = r'Assets\classic_cursor.svg'


def pixel2p(x, y):
    '''Converts pixel coordinates (1080 x 1440) into manim units.'''
    return [
        (x- 720)/1440*FRAME_WIDTH,
        -(y - 540)/1080 *FRAME_HEIGHT,
        0
    ]

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
            font_size=64, font=SANS_SERIF_FONT, weight=LIGHT,
            # stroke_width=0,
            stroke_color=BLACK)
        self.to_edge(UP).shift(UP*0.5)

class HighlightRectangle(BackgroundRectangle):
    def __init__(
        self,
        mobject: Mobject,
        color = BLUE,
        corner_radius: float = 0.1,
        buff: float = 0.05,
        **kwargs
    ):
        super().__init__(mobject, color=color, 
                         stroke_width=0, stroke_opacity=0, fill_opacity=0.4, 
                         buff=buff, corner_radius=corner_radius, **kwargs)
        self.set_z_index(mobject.z_index)
        mobject.set_z_index(mobject.z_index+0.1)
        
class DynamicSplitScreen(VMobject):
    '''Horizontal spliscreen that adapts dynamically to the content.'''
    def __init__(
        self,
        main_color=BLUE,
        side_color=RED,
        buff=SMALL_BUFF*2
    ):
        super().__init__()
        self.mainRect = Rectangle(
            color=main_color, 
            width=FRAME_WIDTH, 
            height=FRAME_HEIGHT, 
            fill_opacity=1, 
            stroke_width=0
        ).set_z_index(0).center()
        self.secondaryRect = Rectangle(
            color=side_color, 
            width=FRAME_WIDTH,
            height= 2 * buff,
            fill_opacity=1,
            stroke_width=0
        ).set_z_index(0).move_to(self.mainRect.get_top(), aligned_edge=DOWN)
        self.mainRect.save_state()
        self.secondaryRect.save_state()

        self.mainObj = None
        self.followMainObj = None
        self.secondaryObj = None
        self.brought_in_ = False
        self.last_shift_ = None
        self.buff_ = buff

        self.mainRect.add_updater(
            lambda r: r.stretch_to_fit_height(
                FRAME_HEIGHT/2 +self.secondaryRect.get_bottom()[1] + 1/1080*FRAME_HEIGHT
                ).move_to([0, -FRAME_HEIGHT/2, 0], aligned_edge=DOWN)
        )
        self.add(self.mainRect, self.secondaryRect)

    def add_main_obj(self, main_obj: VMobject, follow_obj: VMobject = None):
        self.mainObj = main_obj
        self.followMainObj = follow_obj 

    def remove_main_obj(self):
        self.mainObj = None
        self.followMainObj = None

    def add_side_obj(self, secondary_object: VMobject, center_horizontally: bool = True):
        """If the secondary rectangle is outof frame, resizes it and adds the object
        If its in frame, the rectangle is not resized and it is assumed that the the
        object is already in the correct position"""
        self.remove_side_obj()
        if self.brought_in_ == False:
            self.secondaryRect.stretch_to_fit_height(secondary_object.height + 2 * self.buff_)
            self.secondaryRect.move_to(self.mainRect.get_top(), aligned_edge=DOWN)
            if center_horizontally:
                secondary_object.move_to(self.secondaryRect)
            else:
                secondary_object.match_y(self.secondaryRect)
        else:
            self.secondaryRect.move_to([0, +FRAME_HEIGHT/2, 0], aligned_edge=UP)
            self.mainRect.update()
        self.secondaryObj = secondary_object
        self.add(self.secondaryObj)

    def add_empty_side_obj(self, height):
        """Height is intended to be the one of the objects that will appear."""
        self.remove_side_obj()
        self.secondaryRect.stretch_to_fit_height(height + 2 * self.buff_)
        if self.brought_in_ == False:
            self.secondaryRect.move_to(self.mainRect.get_top(), aligned_edge=DOWN)
        else:
            self.secondaryRect.move_to([0, +FRAME_HEIGHT/2, 0], aligned_edge=UP)
            self.mainRect.update()
    
    def remove_side_obj(self):
        if self.secondaryObj is not None:
            self.remove(self.secondaryObj)
            self.secondaryObj = None

    def reset(self):
        self.remove_main_obj()
        self.remove_side_obj()
        self.brought_in_=False
        self.secondaryRect.restore()
        self.mainRect.restore()
        self.mainRect.resume_updating()
        self.last_shift_ = None

    def get_final_mainObj_pos(self):
        return [0, (-self.secondaryRect.height)/2, 0]

    def bring_in(self):
        if not self.brought_in_:
            self.brought_in_=True
            self.secondaryRect.shift(DOWN*self.secondaryRect.height)
            if self.secondaryObj is not None:
                self.secondaryObj.shift(DOWN*self.secondaryRect.height)
            if self.mainObj is not None:
                self.mainObj.shift(DOWN*self.secondaryRect.height/2)
    
    def bring_out(self):
        if self.brought_in_:
            self.brought_in_=False
            self.secondaryRect.shift(UP*self.secondaryRect.height)
            if self.secondaryObj is not None:
                self.secondaryObj.shift(UP*self.secondaryRect.height)
            if self.mainObj is not None:
                self.mainObj.shift(UP*self.secondaryRect.height/2)
    
    def _MoveSecondaryRect(self, direction, **kwargs):
        animations = [
            self.secondaryRect.animate(**kwargs).shift(direction*self.secondaryRect.height),
        ]
        if self.secondaryObj is not None:
            animations.append(
                self.secondaryObj.animate(**kwargs).shift(direction*self.secondaryRect.height),
            )
        if self.mainObj is not None:
            shift = self._get_shift()
            animations.append(
                self.mainObj.animate(**kwargs).shift(direction*shift)
            )
            if self.followMainObj is not None:
                self.followMainObj.shift(direction*shift)

        return AnimationGroup(*animations)
    
    def _get_shift(self) -> float:
        if self.brought_in_:
            shift = self.secondaryRect.height*(0.5 + self.mainObj.get_y()/FRAME_HEIGHT)
            self.last_shift_ = shift
            return shift
        elif self.last_shift_ is not None:
            shift = self.last_shift_
            self.last_shift_ = None
            return shift
        else:
            return self.secondaryRect.height*0.5
    
    def bringIn(self, **kwargs):
        self.brought_in_=True
        return self._MoveSecondaryRect(direction=DOWN, **kwargs)
    
    def bringOut(self, **kwargs):
        self.brought_in_=False
        return self._MoveSecondaryRect(direction=UP, **kwargs)


class Cursor(SVGMobject):
    '''Classic hand cursor for tutorial animations.'''
    def __init__(self, **kwargs):
        super().__init__(file_name=_CURSOR_ICON, height=(24/1080)*FRAME_HEIGHT, **kwargs)

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

_LAPTOP_ICON = r"Assets\laptop_icon.svg"
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

    def _get_spacing(self, n):
        return 0.6*self.Window.height / (n-1) if n > 1 else 0

    def add_inputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs)
        self.InputArrows = VGroup(Arrow(ORIGIN, RIGHT*arrow_length, color=BLUE, stroke_width=6) for _ in range(n_inputs))
        self.InputArrows.arrange(DOWN, buff=spacing).next_to(self.Window, LEFT, buff=0)
        self.InputLabels = VGroup(
            Text(l, color=BLUE, font=CODE_FONT)
            if isinstance(l, str) else l  for l in labels
        )
        for i in range(n_inputs):
            self.InputLabels[i].next_to(self.InputArrows[i], LEFT, buff=buff)
        
        self.add(self.InputArrows, self.InputLabels)

    def add_outputs(self, *labels: VMobject | str, arrow_length=1.5, buff=SMALL_BUFF):
        n_inputs = len(labels)
        spacing = self._get_spacing(n_inputs)
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
            include_outer_lines=True,
            line_config={'stroke_width':7, 'color':color}
        )
    
    def get_lines(self) -> VGroup:
        return self.get_horizontal_lines() + self.get_vertical_lines()
