'''Generic mooc utils'''

__all__ = [
    "FRAME_HEIGHT", "FRAME_WIDTH", "ASPECT_RATIO", "CODE_FONT",
    "HALF_SCREEN_LEFT", "HALF_SCREEN_RIGHT",
    "pixel2p", "Cursor", "DynamicSplitScreen", "HighlightRectangle",
    "CustomDecimalNumber"
]

from manim import *
import manimpango

FRAME_HEIGHT = 10.66  # In 4:3 frame height is 10.66, not 8!
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO
HALF_SCREEN_LEFT = [-FRAME_WIDTH/4, 0, 0]
HALF_SCREEN_RIGHT = [+FRAME_WIDTH/4, 0, 0]

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
        self.secondaryObj = None
        self.brought_in_ = False
        self.buff_ = buff

        self.mainRect.add_updater(
            lambda r: r.stretch_to_fit_height(
                FRAME_HEIGHT/2 +self.secondaryRect.get_bottom()[1] + 1/1080*FRAME_HEIGHT
                ).move_to([0, -FRAME_HEIGHT/2, 0], aligned_edge=DOWN)
        )
        self.add(self.mainRect, self.secondaryRect)

    def add_main_obj(self, main_obj: VMobject):
        self.mainObj = main_obj

    def remove_main_obj(self):
        self.mainObj = None

    def add_side_obj(self, secondary_object: VMobject):
        self.remove_side_obj()
        self.secondaryRect.stretch_to_fit_height(secondary_object.height + 2 * self.buff_)
        if self.brought_in_ == False:
            self.secondaryRect.move_to(self.mainRect.get_top(), aligned_edge=DOWN)
        else:
            self.secondaryRect.move_to([0, +FRAME_HEIGHT/2, 0], aligned_edge=UP)
            self.mainRect.update()
        self.secondaryObj = secondary_object
        secondary_object.move_to(self.secondaryRect)
        self.add(self.secondaryObj)
    
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
    
    def bringIn(self, **kwargs):
        self.brought_in_=True
        if self.mainObj is not None:
            return AnimationGroup(
                self.secondaryRect.animate(**kwargs).shift(DOWN*self.secondaryRect.height),
                self.secondaryObj.animate(**kwargs).shift(DOWN*self.secondaryRect.height),
                self.mainObj.animate(**kwargs).shift(DOWN*self.secondaryRect.height/2)
            )
        else:
            return AnimationGroup(
                self.secondaryRect.animate(**kwargs).shift(DOWN*self.secondaryRect.height),
                self.secondaryObj.animate(**kwargs).shift(DOWN*self.secondaryRect.height),
            )
    
    def bringOut(self, **kwargs):
        self.brought_in_=False
        if self.mainObj is not None:
            return AnimationGroup(
                self.secondaryRect.animate(**kwargs).shift(UP*self.secondaryRect.height),
                self.secondaryObj.animate(**kwargs).shift(UP*self.secondaryRect.height),
                self.mainObj.animate(**kwargs).shift(UP*self.secondaryRect.height/2)
            )
        else:
            return AnimationGroup(
                self.secondaryRect.animate(**kwargs).shift(UP*self.secondaryRect.height),
                self.secondaryObj.animate(**kwargs).shift(UP*self.secondaryRect.height),
            )


class Cursor(SVGMobject):
    '''Classic hand cursor for tutorial animations.'''
    def __init__(self, **kwargs):
        super().__init__(file_name=_CURSOR_ICON, height=(24/1080)*FRAME_HEIGHT, **kwargs)

    def Click(self):
        return self.animate(rate_func=there_and_back, run_time=0.1).scale(0.8)
    
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