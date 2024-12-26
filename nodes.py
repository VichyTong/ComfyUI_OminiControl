import sys
import os
import importlib

# sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

class Kiki_Omini_Subject:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Subject'

    CATEGORY = "Runninghub/Omini"
    DESCRIPTION = "Ominicontrol subject node"

    def run(self, subject_image, prompt, seed):
        import ComfyUI_RH_OminiControl.rh_omini_subject as ros
        img = ros.run(subject_image, prompt, seed)
        return (img, )
    
class Kiki_Omini_Spatial:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "condition_type": (["canny", "depth", "coloring", "deblurring"], ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Spatial'

    CATEGORY = "Runninghub/Omini"
    DESCRIPTION = "Ominicontrol spatial node"

    def run(self, ref_image, prompt, condition_type, seed):
        import ComfyUI_RH_OminiControl.rh_omini_spatial as rosp
        img = rosp.run(ref_image, prompt, condition_type, seed)
        return (img, )
    
class Kiki_Omini_Fill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ori_image": ("IMAGE",),
                "mask": ("MASK", ),
                "prompt": ("STRING", {"multiline": True,
                                      "default": ''}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    TITLE = 'OminiControl Fill'

    CATEGORY = "Runninghub/Omini"
    DESCRIPTION = "Ominicontrol fill node"

    def run(self, ori_image, mask, prompt, seed):
        import ComfyUI_RH_OminiControl.rh_omini_fill as rof
        img = rof.run(ori_image, mask, prompt, seed)
        return (img, )

NODE_CLASS_MAPPINGS = {
    "RunningHub_Omini_Subject": Kiki_Omini_Subject,
    "RunningHub_Omini_Spatial": Kiki_Omini_Spatial,
    "RunningHub_Omini_Fill": Kiki_Omini_Fill,
}
