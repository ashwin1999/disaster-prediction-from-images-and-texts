import numpy as np


def make_text_prediction(pred, multi=False):
    if multi:
        type1 = []
        for val in pred:
            if val > 0.5:
                type1.append('Disaster')
            else:
                type1.append('Normal')
        return type1
    else:
        type2 = 'Normal'
        if pred > 0.5:
            type2 = 'Disaster'
        return type2


def make_image_prediction(pred):
    mapping = {
        0: 'Damaged Infrastructure',
        1: 'Drought',
        2: 'Earthquake',
        3: 'Injured Human',
        4: 'Land Slide',
        5: 'Safe Forest',
        6: 'Safe Human',
        7: 'Safe Infrastructure',
        8: 'Safe Water',
        9: 'Urban Fire',
        10: 'Water Disaster',
        11: 'Wild Fire'
    }

    return mapping[np.argmax(pred[0])]
