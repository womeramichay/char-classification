#
# **** Character Classifier Task ****
#
# In this project we face the problem a classifying characters, given their B&W images.
# The train set will consist of characters from a given set of fonts, while the test set
# is created using different fonts.
#
# Implement a classifier that is optimized for the "Digits" and "ASCII" tests.
# 

import itertools
import random
import string
import numpy as np
import pandas
from collections import OrderedDict
from PIL import ImageFont, ImageDraw, Image
from matplotlib.font_manager import findSystemFonts


FONT_SIZE = 24
FONT_PATHS = sorted(findSystemFonts(fontpaths=None, fontext='ttf'))
FONTS = [ImageFont.truetype(fpath, size=FONT_SIZE) for fpath in FONT_PATHS]


def draw_text(text, font=None):
    image = Image.new("RGB", (30 * len(text), 30), "white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black", spacing=0, stroke_fill="black")
    image = image.point(lambda p: 255 if p > 200 else 0)
    return image


def as_binary_array(image):
    return np.asarray(image)[:, :, 0]


class CharGenerator:
    IMG_SIZE = 50

    def __init__(self, chars: set, fonts=None, shuffle=True):
        self.chars = sorted(list(chars))
        self.fonts = fonts
        self.size = len(chars) * (len(fonts) if fonts else 1)
        self.index = 0
        self.fonts_and_chars = None
        self.shuffle = shuffle

    def __iter__(self):
        self.index = 0
        self.fonts_and_chars = list(itertools.product(self.fonts, self.chars))
        if self.shuffle:
            random.seed(0)
            random.shuffle(self.fonts_and_chars)
        return self

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration()
        font, char = self.fonts_and_chars[self.index]
        image = draw_text(char, font)
        image = as_binary_array(image)
        self.index += 1
        return char, image


class CharClassifier:

    def train(self, generator: CharGenerator):
        raise NotImplementedError()

    def predict(self, image):
        raise NotImplementedError()


class RandomClassifier(CharClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chars = None

    def train(self, generator):
        self.chars = list(generator.chars)

    def predict(self, image):
        return random.sample(self.chars, 1)[0]


def calc_accuracy(classifier: CharClassifier, test_set_generator: CharGenerator):
    is_correct = [classifier.predict(img) == char for char, img in test_set_generator]
    return sum(is_correct) / len(is_correct)


class Test:
    def __init__(self, name, chars: set, train_fonts, test_fonts):
        self.name = name
        self.train_set = CharGenerator(chars=chars, fonts=train_fonts)
        self.test_set = CharGenerator(chars=chars, fonts=test_fonts)


TESTS = [
    Test(name="Digits", chars=set(string.digits), train_fonts=FONTS[:10], test_fonts=FONTS[10:30]),
    Test(name="Digits: Small Train Set", chars=set(string.digits), train_fonts=FONTS[:1], test_fonts=FONTS[10:30]),
    Test(name="ASCII", chars=set(string.ascii_letters), train_fonts=FONTS[:10], test_fonts=FONTS[10:30]),
    Test(name="ASCII: Small Train Set", chars=set(string.ascii_letters), train_fonts=FONTS[:1],
         test_fonts=FONTS[10:30]),
]

if __name__ == '__main__':

    classifier_classes = [
        RandomClassifier
    ]

    results = []
    for cls in classifier_classes:
        for test in TESTS:
            classifier = cls()
            classifier.train(test.train_set)
            results.append(OrderedDict([
                ("Classifier", cls.__name__),
                ("Test", test.name),
                ("Train Set Size", test.train_set.size),
                ("Test Set Size", test.test_set.size),
                ("Accuracy", round(calc_accuracy(classifier, test.test_set), 3)),
            ]))

    df = pandas.DataFrame(results)
    pandas.set_option("display.width", 400)
    pandas.set_option("display.max_columns", 20)
    print(df)
