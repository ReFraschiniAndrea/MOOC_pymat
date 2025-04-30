'''Generic mooc utils'''

__all__ = [
    "FRAME_HEIGHT", "FRAME_WIDTH", "ASPECT_RATIO",
    "CODE_FONT", "pixel2p", "Cursor"
]

from manim import *
import manimpango

FRAME_HEIGHT = 10.66  # In 4:3 frame height is 10.66, not 8!
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO

CODE_FONT = 'Aptos Mono'
try:
    manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos-Mono.ttf")
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
