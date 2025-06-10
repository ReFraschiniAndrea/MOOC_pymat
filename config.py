from manim import Scene, ThreeDScene
from manim_slides.config import BaseSlideConfig
from manim_slides.slide.manim import Slide
from manim_slides.slide.base import BaseSlide
from pydantic import model_validator
from textwrap import dedent
from typing import Any

__all__ = [
    "MOOCSlide", "ThreeDMOOCSlide"
]

class MOOCSlideConfig(BaseSlideConfig):  # type: ignore
    """Base class for slide config."""
    @model_validator(mode="after")
    def apply_dedent_notes(
        self,
    ) -> "MOOCSlideConfig":
        if self.dedent_notes:
            if not self.notes.startswith("\n"):
                self.notes = "            " + self.notes
            self.notes = dedent(self.notes).strip("\n")

        return self
    
class MOOCSlide(Slide):
    def __init__(self):
        super().__init__()
        self.wait_time_between_slides = 0.05
        self.skip_reversing = True

    @MOOCSlideConfig.wrapper("base_slide_config")
    def next_slide(
        self,
        *args: Any,
        base_slide_config: MOOCSlideConfig,
        **kwargs: Any,
    ) -> None:
        Scene.next_section(
            self,
            *args,
            skip_animations=base_slide_config.skip_animations | self._skip_animations,
            **kwargs,
        )
        BaseSlide.next_slide.__wrapped__(
            self,
            base_slide_config=base_slide_config,
        )

class ThreeDMOOCSlide(MOOCSlide, ThreeDScene):
    pass