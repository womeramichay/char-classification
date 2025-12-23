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


class MyClassifier(CharClassifier):
    """
    Prototype-based character classifier with robust preprocessing
    and light, controlled augmentation.
    """

    def __init__(self):
        self.prototypes = None  # dict: char -> mean feature vector

    # =========================
    # Preprocessing
    # =========================
    @staticmethod
    def _to_features(img: np.ndarray, target: int = 15) -> np.ndarray:
        """
        img: 2D uint8 array with values {0,255}
        Returns a flattened feature vector.
        """

        # 1) binarize: ink=1, background=0
        x = (img == 0).astype(np.float32)

        # 2) crop to foreground bounding box
        ys, xs = np.where(x > 0)
        if len(xs) > 0:
            x = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        # 3) pad to square
        h, w = x.shape
        s = max(h, w)
        pad_y, pad_x = s - h, s - w
        x = np.pad(
            x,
            ((pad_y // 2, pad_y - pad_y // 2),
             (pad_x // 2, pad_x - pad_x // 2)),
            mode="constant",
            constant_values=0.0
        )

        # 4) downsample via block-mean pooling
        step = max(1, s // target)
        x = x[: step * target, : step * target]
        x = x.reshape(target, step, target, step).mean(axis=(1, 3))

        return x.reshape(-1)

    # =========================
    # Augmentations (binary-safe)
    # =========================
    @staticmethod
    def _shift_binary(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = img.shape
        out = np.full((h, w), 255, dtype=img.dtype)

        y0s, y1s = max(0, -dy), min(h, h - dy)
        x0s, x1s = max(0, -dx), min(w, w - dx)
        y0d, x0d = max(0, dy), max(0, dx)

        out[y0d:y0d + (y1s - y0s), x0d:x0d + (x1s - x0s)] = img[y0s:y1s, x0s:x1s]
        return out

    @staticmethod
    def _rotate_binary(img: np.ndarray, degrees: float) -> np.ndarray:
        pil = Image.fromarray(img)
        pil = pil.rotate(degrees, resample=Image.NEAREST, fillcolor=255)
        arr = np.array(pil, dtype=np.uint8)
        return np.where(arr < 128, 0, 255).astype(np.uint8)

    @staticmethod
    def _saltpepper_binary(img: np.ndarray, p_flip: float = 0.003) -> np.ndarray:
        out = img.copy()
        h, w = out.shape
        n = int(h * w * p_flip)
        if n <= 0:
            return out
        ys = np.random.randint(0, h, n)
        xs = np.random.randint(0, w, n)
        out[ys, xs] = np.where(out[ys, xs] == 0, 255, 0)
        return out

    # =========================
    # Training
    # =========================
    def train(self, generator: CharGenerator):
        k = 2               # number of augmented copies per image
        max_shift = 2
        max_rot = 10        # degrees

        by_char = {c: [] for c in generator.chars}

        transforms = ["shift", "rotate", "noise"]
        weights = [0.55, 0.35, 0.10]  # exactly ONE transform per augmentation

        for char, img in generator:
            # original
            by_char[char].append(self._to_features(img))

            # augmented samples
            for _ in range(k):
                t = random.choices(transforms, weights=weights, k=1)[0]

                if t == "shift":
                    aug = self._shift_binary(
                        img,
                        random.randint(-max_shift, max_shift),
                        random.randint(-max_shift, max_shift),
                    )
                elif t == "rotate":
                    aug = self._rotate_binary(
                        img,
                        random.uniform(-max_rot, max_rot),
                    )
                else:
                    aug = self._saltpepper_binary(img)

                by_char[char].append(self._to_features(aug))

        # prototype per class
        self.prototypes = {
            c: np.mean(np.stack(v, axis=0), axis=0)
            for c, v in by_char.items()
        }

    # =========================
    # Prediction
    # =========================
    def predict(self, image: np.ndarray):
        f = self._to_features(image)
        best_char, best_dist = None, float("inf")
        for c, p in self.prototypes.items():
            d = np.sum((f - p) ** 2)
            if d < best_dist:
                best_char, best_dist = c, d
        return best_char



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
