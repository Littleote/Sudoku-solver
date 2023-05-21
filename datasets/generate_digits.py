# -*- coding: utf-8 -*-
"""
@author: david
"""

from cairosvg import svg2png
import numpy as np

SCALE = 28
FILE = "digits{scale}/{id1}{key1}{id2}{key2}.png"
SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" width="{scale}" height="{scale}">
        <rect width="100%" height="100%" fill="#ffffff" />
        <text x="50%" y="50%" dominant-baseline="middle" font-weight="{weight}" text-anchor="middle" font-size="{size}" font-family="{family}">{text}</text>
    </svg>
"""


def format_info(*, number, font_family, font_size, font_weight, scale):
    """
    Set format to svg code and file output name
    """
    svg = SVG.format(text=number, family=font_family,
                     size=font_size / 100 * SCALE, weight=font_weight,
                     scale=scale)
    file = FILE.format(
        id1=number, key1=font_weight[0], id2=font_size, key2=font_family,
        scale=scale)
    return svg, file


NUMBERS = [i for i in range(10)]
FONT_FIMILIES = [
    'arial',
    'verdana',
    'tahoma',
    'trebuchet MS',
    'Times New Roman',
    'Georgia',
    'Garamond',
    'Courier New',
    'Brush Scipt MT'
]
FONT_SIZES = np.arange(50, 120, 5)
FONT_WEIGHT = ['normal', 'bold']

def _main(scale=SCALE):
    print(f"Generating for size {scale}.")
    for n in NUMBERS:
        print(f"{n*10}%")
        for f in FONT_FIMILIES:
            for s in FONT_SIZES:
                for w in FONT_WEIGHT:
                    svg, file = format_info(
                        number=n, font_family=f,
                        font_size=s, font_weight=w,
                        scale=scale)
                    svg2png(bytestring=svg, write_to=file)
                    
if __name__ == "__main__":
    _main()
