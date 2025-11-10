import os
import manimpango

from .config import *
from .Generic_mooc_utils import *

# Look for a MikTex installation in AppData\Local
# if os.path.exists("C:\\Users"):
#     for user in os.listdir("C:\\Users"):
#         candidate_path = os.path.join("C:\\Users", user, r"AppData\Local\Programs\MiKTeX\miktex\bin\x64")
#         if os.path.exists(candidate_path):
#             os.environ["PATH"] = candidate_path + ";" + os.environ["PATH"]
#             break

manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos-Mono.ttf")
manimpango.register_font(r"Assets\Fonts\Microsoft Aptos Fonts\Aptos.ttf")

del os
del manimpango
del config