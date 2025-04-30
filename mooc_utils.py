'''Utils hared by different MOOC presentations.'''

from manim import *
import manimpango

CODE_FONT = 'Aptos Mono'
try:
    manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos-Mono.ttf")
except:
    print('warning, unable to find font. falling back to monospace.')
    CODE_FONT = 'Monospace'

# Frame constants
FRAME_HEIGHT = 10.66  # In 4:3 frame height is 10.66, not 8!
ASPECT_RATIO = 4/3
FRAME_WIDTH = FRAME_HEIGHT * ASPECT_RATIO
# Colab constants
COLAB_LIGHTGRAY = "#f7f7f7"
COLAB_GRAY = "#eeeeee"
COLAB_DARKGRAY ="#424242"
COLAB_LEFT_BUFF = 127/1440 * FRAME_WIDTH
COLAB_GUTTER_WIDTH = 50/1080 *FRAME_HEIGHT
COLAB_BLOCK_WIDTH = 1300/1440 * FRAME_WIDTH
COLAB_FONT_SIZE = 12
COLAB_BUTTON_RADIUS = (23/1080*FRAME_HEIGHT)/2
COLAB_SURROUND_CODE_BUFF = 17/1080*FRAME_HEIGHT
COLAB_GUTTER_TO_TEXT_BUFF = 10/1440 * FRAME_WIDTH
# logo constants
_PYTHON_LOGO = r'Assets\python_logo.png'
_MATLAB_LOGO = r'Assets\matlab_logo.png'
_LOGO_SHIFT_BUFF = 0.5
_CURSOR_ICON = r'Assets\classic_cursor.svg'

def pixel2p(x, y):
    return [
        (x- 720)/1440*FRAME_WIDTH,
        -(y - 540)/1080 *FRAME_HEIGHT,
        0
    ]

class HalfScreenRectangle(Rectangle):
    def __init__(self, color, stroke_width=0, stroke_color=BLACK):
        super().__init__(height=FRAME_HEIGHT,
                         width=FRAME_HEIGHT*ASPECT_RATIO /2, 
                         color=color, fill_opacity=1, 
                         stroke_width=stroke_width, stroke_color=stroke_color)
        
class HighlightRectangle(BackgroundRectangle):
    def __init__(self, mobject: Mobject, color = BLUE, corner_radius =0.1, buff = 0.05, **kwargs):
        super().__init__(mobject, color=color, 
                         stroke_width=0, stroke_opacity=0, fill_opacity=1, 
                         buff=buff, corner_radius=corner_radius, **kwargs)
        self.set_z_index(mobject.z_index - 1)
        
class DynamicSplitScreen(VMobject):
    def __init__(self, main_color=BLUE, side_color=RED, buff=SMALL_BUFF*2):
        super().__init__()
        self.mainRect = Rectangle(
            color=main_color, 
            width=FRAME_WIDTH, 
            height=FRAME_HEIGHT, 
            fill_opacity=1, 
            stroke_width=0
        ).center()
        self.secondaryRect = Rectangle(
            color=side_color, 
            width=FRAME_WIDTH,
            height= 2 * buff,
            fill_opacity=1,
            stroke_width=0
        ).move_to(self.mainRect.get_top(), aligned_edge=DOWN)
        self.mainObj = None
        self.mainWindow = None
        self.secondaryObj = None
        self.secondaryWindow = None
        self.brought_in_ = False
        self.buff = buff

        self.mainRect.add_updater(
            lambda r: r.stretch_to_fit_height(
                FRAME_HEIGHT/2 +self.secondaryRect.get_bottom()[1]
                ).move_to([0, -FRAME_HEIGHT/2, 0], aligned_edge=DOWN)
        )
        self.add(self.mainRect, self.secondaryRect)

    def add_main_obj(self, main_obj: VMobject):
        self.mainObj = main_obj
        self.mainObj.move_to(self.mainRect)
        # self.mainObj.add_updater(
        #     lambda m: m.move_to((self.mainRect.get_top() +self.mainRect.get_bottom())/2)
        # )
        # self.mainWindow = VGroup(self.mainRect, self.mainObj)

    def remove_main_obj(self):
        self.mainObj = None

    def add_side_obj(self, secondary_object: VMobject):
        self.secondaryRect.stretch_to_fit_height(secondary_object.height + 2 * self.buff)
        if self.brought_in_ == False:
            self.secondaryRect.move_to(self.mainRect.get_top(), aligned_edge=DOWN)
        else:
            self.secondaryRect.move_to([0, -FRAME_HEIGHT/2, 0], aligned_edge=UP)
            self.mainRect.update()
        secondary_object.move_to(self.secondaryRect)
        self.secondaryObj = secondary_object
        self.secondaryWindow = Group(self.secondaryRect, secondary_object)

    def bringIn(self, **kwargs):
        self.brought_in_=True
        if self.mainObj is not None:
            return AnimationGroup(
                self.secondaryWindow.animate(**kwargs).shift(DOWN*self.secondaryRect.height),
                self.mainObj.animate(**kwargs).shift(DOWN*self.secondaryRect.height/2)
            )
        else:
            return self.secondaryWindow.animate(**kwargs).shift(DOWN*self.secondaryRect.height)

    
    def hard_bring_in(self):
        if self.brought_in_ == False:
            self.secondaryWindow.shift(DOWN*self.secondaryRect.height),
            self.mainObj.shift(DOWN*self.secondaryRect.height/2)
    
    def bringOut(self, **kwargs):
        self.brought_in_ = False
        if self.mainObj is not None:
            return AnimationGroup(
                self.secondaryWindow.animate(**kwargs).shift(UP*self.secondaryRect.height),
                self.mainObj.animate(**kwargs).shift(UP*self.secondaryRect.height/2)
            )
        else:
            return self.secondaryWindow.animate(**kwargs).shift(UP*self.secondaryRect.height)

class CustomCode(Mobject):
    def __init__(self, code_string, language=None, formatter_style=None, 
                 background_config=None, paragraph_config=None,
                 include_logo=False, logo_mobj: ImageMobject=None, logo_pos=UP, 
                 logo_buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER):
        super().__init__()
        self.code_string = code_string
        self.codeMobject = Code(
            code_string=code_string,
            language=language,
            formatter_style=formatter_style,
            background="rectangle",
            add_line_numbers=False,
            background_config=background_config,
            paragraph_config=paragraph_config
        )
        self.window = self.codeMobject[0]
        self.code = self.codeMobject[1]
        self.add(self.codeMobject)
        self.include_logo = include_logo
        if include_logo:
            self.logo = logo_mobj.next_to(self.codeMobject, logo_pos, buff=logo_buff)
            if (logo_pos == UP).all():
                self.logo.align_to(self.window, LEFT).shift(RIGHT*_LOGO_SHIFT_BUFF)
            elif (logo_pos == LEFT).all():
                 self.logo.align_to(self.window, UP).shift(DOWN*_LOGO_SHIFT_BUFF)
            self.add(self.logo)

    def scroll(self, n_lines_fade, n_lines_shift=None, shown_lines=None, **kwargs):
        if n_lines_shift is None:
            n_lines_shift = n_lines_fade
        if shown_lines is None:
            shown_lines = len(self.code)
        assert(shown_lines>=n_lines_fade)
        # interline = self.code[0][0].get_y() - self.code[1].get_y()
        interline = self.code.height / len(self.code)
        srcroll_shift = n_lines_shift*interline*UP

        fade = FadeOut(self.code[:n_lines_fade], shift=srcroll_shift, **kwargs)
        shift = self.code[n_lines_fade:shown_lines].animate(**kwargs).shift(srcroll_shift)
        self.code[shown_lines:].shift(srcroll_shift)

        return AnimationGroup(fade, shift, lag_ratio=0)
    
    def typeLetterbyLetter(self, lines=None, lag_ratio=1):
        if lines is None:
            lines = range(len(self.code))
        anims = []
        for i in lines:
            if self.code[i].family_members_with_points():
                anims.append(AddTextLetterByLetter(self.code[i], rate_func=linear, time_per_char=0.0005))
        return AnimationGroup(*anims, lag_ratio=lag_ratio)

        

class ColabCode(CustomCode):
    def __init__(self, code, include_logo=False, logo_pos=UP, logo_buff =DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
                 font_size=24, line_spacing=0.5, corner_radius=0.2, fill_opacity=1, bckgr_buff=MED_SMALL_BUFF):
        if include_logo:
            logo = ImageMobject(_PYTHON_LOGO).scale(0.5)
        else:
            logo = None
        super().__init__(
            code_string=code,
            language='python',
            formatter_style='paraiso-light',
            include_logo=include_logo,
            logo_mobj=logo,
            logo_buff=logo_buff,
            logo_pos=logo_pos,
            background_config={
                "buff": 0.3,
                "fill_color": COLAB_LIGHTGRAY,
                "stroke_color": BLACK,
                "corner_radius":corner_radius, # sharp corners
                "stroke_width": 0,  # no outline
                "fill_opacity": fill_opacity, 
                "buff": bckgr_buff
            },
            paragraph_config={
                "font": CODE_FONT,
                "font_size": font_size,
                "line_spacing": line_spacing,
                "disable_ligatures": True,
                "color":BLACK   # color of the line numbers
            }
        )

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
        self.colabCode = ColabCode(code, include_logo=False, 
                         font_size=COLAB_FONT_SIZE,
                         line_spacing=0.5, corner_radius=0, fill_opacity=1,
                         bckgr_buff=COLAB_SURROUND_CODE_BUFF)
        target_height = self.colabCode.window.height if code != '' else 0.4766081925925926
        self.colabCode.window.become(Rectangle(fill_color=COLAB_LIGHTGRAY, stroke_width=0, fill_opacity=1, width=COLAB_BLOCK_WIDTH, height = target_height))
        self.gutter = self._create_Gutter().next_to(self.colabCode.window, LEFT, buff=0)
        self.playButton = self._create_PlayButton().move_to(self.gutter.get_center()).align_to(self.gutter, UP).shift(DOWN*0.1)
        # be careful if empty
        if code != '':
            self.colabCode.code.align_to(self.colabCode.window, LEFT).shift(RIGHT*COLAB_GUTTER_TO_TEXT_BUFF)
        self.output = None
        self.outputWindow = None

        self.add(self.colabCode)
        self.add(self.gutter)
        self.add(self.playButton)

    def _create_PlayButton(self):
        button = Circle(
            color=COLAB_DARKGRAY, 
            radius=COLAB_BUTTON_RADIUS,
            fill_opacity=1)
        tri = Triangle(
            color=COLAB_LIGHTGRAY,
            radius=COLAB_BUTTON_RADIUS*0.7,
            fill_opacity=1,
            stroke_width=0
            ).rotate(-PI/2).move_to(button).shift(RIGHT*COLAB_BUTTON_RADIUS*0.7/4)
        return VGroup(button, tri)
    
    def _create_Gutter(self):
        return Rectangle(
            color=COLAB_GRAY, 
            width=COLAB_GUTTER_WIDTH, 
            height=self.colabCode.window.height,
            fill_opacity=1,
            stroke_width=0)
        
    def add_output(self, output:str | Mobject):
        if isinstance(output, str):
            self.output = ColabBlockOutputText(output)
        else:
            self.output=output
        self.outputWindow = SurroundingRectangle(
            self.output,
            color=WHITE,
            fill_opacity=1,
            stroke_width=0,
            buff=COLAB_SURROUND_CODE_BUFF)
        self.outputWindow.become(Rectangle(
            width=COLAB_BLOCK_WIDTH + COLAB_GUTTER_WIDTH, 
            height=self.outputWindow.height,
            color=WHITE,
            fill_opacity=1,
            stroke_width=0))
        self.output.move_to(self.outputWindow)
        Group(self.outputWindow, self.output).next_to(VGroup(self.colabCode.window, self.gutter), DOWN, buff=0)
        self.output.align_to(self.colabCode.code, LEFT)

        self.add(self.outputWindow)
        self.add(self.output)


    def run(self):
        self.cursor = Cursor().move_to(self.playButton)
        self.add(self.cursor)
        # if self.output is not None and self.outputWindow is not None:
        return Succession(
            GrowFromCenter(self.cursor),
            AnimationGroup(
                self.cursor.click(),
                FadeIn(self.outputWindow, self.output, run_time=0),
                lag_ratio=0.5),
            lag_ratio=1
        )
        # else:
        # return Succession(
        #     GrowFromCenter(self.cursor),
        #     self.cursor.click()
        # )

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
            self.remove(self.cells[-1])
            self.cells.pop()
    
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


class Cursor(SVGMobject):
    def __init__(self, **kwargs):
        super().__init__(file_name=_CURSOR_ICON, height=(24/1080)*FRAME_HEIGHT, **kwargs)

    def click(self):
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
    
    
class MatlabCode(CustomCode):
    def __init__(self, code, include_logo=False, logo_pos=UP, logo_buff =DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
                 font_size=24, line_spacing=0.5, fill_opacity=1, stroke_width=0.5):
        if include_logo:
            logo = ImageMobject(_MATLAB_LOGO).scale(0.5)
        else:
            logo = None
        super().__init__(
            code_string=code,
            language='matlab',
            formatter_style='sas',
            include_logo=include_logo,
            logo_mobj=logo,
            logo_buff=logo_buff,
            logo_pos=logo_pos,
            background_config={
                "buff": 0.3,
                "fill_color": WHITE,
                "stroke_color": BLACK,
                "corner_radius": 0.2, # sharp corners
                "stroke_width": stroke_width,  # no outline
                "fill_opacity": fill_opacity, 
            },
            paragraph_config={
                "font": CODE_FONT,
                "font_size": font_size,
                "line_spacing": line_spacing,
                "disable_ligatures": True,
                "color": BLACK   # color of the line numbers
            }
        )

    def into_matlab(self, matlab_env: 'MatlabEnv', **kwargs):
        target = MatlabCode(self.code_string, font_size=COLAB_FONT_SIZE, fill_opacity=0, stroke_width=0)
        matlab_env.add_cell(target)
        return AnimationGroup(
            ReplacementTransform(self.window, target.window, **kwargs),
            ReplacementTransform(self.code, target.code, **kwargs),
            FadeIn(matlab_env, **kwargs)
        )


class MatlabEnv(Mobject):
    def __init__(self, background=None):
        super().__init__()
        self.TOP_LEFT_CORNER = 0
        self.OUTPUT_TOP_LEFT = 0
        self.env_image = ImageMobject(background).scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1) # background
        self.add(self.env_image)

def CodeTransform(code1: CustomCode, code2: CustomCode, **kwargs):
    assert(code1.include_logo == code2.include_logo)
    if code1.include_logo:
        return AnimationGroup(
            Transform(code1.codeMobject, code2.codeMobject, **kwargs),
            code1.logo.animate(**kwargs).become(code2.logo)
        )
    else:
        return Transform(code1.codeMobject, code2.codeMobject, **kwargs)