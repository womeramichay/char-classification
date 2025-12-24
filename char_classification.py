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
    Prototype-based classifier:
    - preprocess: binarize to 0/1, crop bbox, pad to square, block-mean downsample
    - training: build per-class prototype (mean feature vector)
    - augmentation: k augmented copies per sample; each copy applies exactly ONE transform
    - prediction: nearest prototype via L2 or cosine distance
    """

    def __init__(
        self,
        # feature extraction
        target: int = 15,

        # augmentation
        k: int = 3,
        max_shift: int = 2,
        max_rot: float = 12.0,
        p_flip: float = 0.0,                 # default OFF
        weights=(0.45, 0.55, 0.0),           # (shift, rotate, noise) default OFF

        # distance metric
        distance: str = "l2",                # "l2" or "cosine"

        # reproducibility
        seed: int | None = 42,
    ):
        self.target = int(target)

        self.k = int(k)
        self.max_shift = int(max_shift)
        self.max_rot = float(max_rot)
        self.p_flip = float(p_flip)
        self.weights = tuple(weights)

        self.distance = distance
        if self.distance not in ("l2", "cosine"):
            raise ValueError("distance must be 'l2' or 'cosine'")

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.prototypes = None  # dict: char -> prototype vector

    # =========================
    # Preprocessing
    # =========================
    def _to_features(self, img: np.ndarray) -> np.ndarray:
        """
        img: 2D uint8 array with values {0,255}
        Returns a flattened feature vector of length target*target.
        """

        # 1) binarize: ink=1, background=0 (in this task ink pixels are 0)
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

        # 4) ensure minimum size before downsampling
        h = x.shape[0]
        if h < self.target:
            pad = self.target - h
            x = np.pad(
                x,
                ((pad // 2, pad - pad // 2),
                 (pad // 2, pad - pad // 2)),
                mode="constant",
                constant_values=0.0
            )

        # 5) downsample via block-mean pooling to (target x target)
        s = x.shape[0]
        step = max(1, s // self.target)

        # make divisible by target
        x = x[: step * self.target, : step * self.target]

        # reshape + mean pool
        x = x.reshape(self.target, step, self.target, step).mean(axis=(1, 3))

        return x.reshape(-1)

    # =========================
    # Distance
    # =========================
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return 1.0 - float(np.dot(a, b) / denom)

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
        if p_flip <= 0:
            return img
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
        by_char = {c: [] for c in generator.chars}
        transforms = ["shift", "rotate", "noise"]

        for char, img in generator:
            # original
            by_char[char].append(self._to_features(img))

            # augmented copies: exactly ONE transform per copy
            for _ in range(self.k):
                t = random.choices(transforms, weights=self.weights, k=1)[0]

                if t == "shift":
                    aug = self._shift_binary(
                        img,
                        random.randint(-self.max_shift, self.max_shift),
                        random.randint(-self.max_shift, self.max_shift),
                    )
                elif t == "rotate":
                    aug = self._rotate_binary(
                        img,
                        random.uniform(-self.max_rot, self.max_rot),
                    )
                else:
                    aug = self._saltpepper_binary(img, p_flip=self.p_flip)

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
            if self.distance == "cosine":
                d = self._cosine_distance(f, p)
            else:
                d = float(np.sum((f - p) ** 2))

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
    Test(name="ASCII: Small Train Set", chars=set(string.ascii_letters), train_fonts=FONTS[:1], test_fonts=FONTS[10:30]),
]


if __name__ == '__main__':

    classifier_classes = [
        RandomClassifier,
        MyClassifier,
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
