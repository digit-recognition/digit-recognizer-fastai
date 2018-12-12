from recognizer import recognizer

from fastai.imports import *

if __name__ == "__main__":
    #PATH = Path('./data')
    PATH = Path('.')
    IMG_SIZE = 28

    recognizer = recognizer.Recognizer()
    recognizer.recognize(PATH, IMG_SIZE)
    # recognizer.recognize2(PATH, IMG_SIZE)
