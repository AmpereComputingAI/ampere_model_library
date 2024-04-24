# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
from random import randint, seed
from utils.helpers import Dataset


class StableDiffusion(Dataset):

    def __init__(self):
        self._idx = 0
        self.available_instances = 100000

    def get_input(self):
        adjectives = ["big", "small", "thin", "wide", "blonde", "pale"]
        nouns = ["dog", "cat", "horse", "astronaut", "human", "robot"]
        actions = ["sings", "rides a triceratop", "rides a horse", "eats a burger", "washes clothes", "looks at hands"]
        seed(42)

        a = adjectives[randint(0, len(adjectives) - 1)] + " "
        b = nouns[randint(0, len(nouns) - 1)] + " " + actions[randint(0, len(actions) - 1)]
        return a + b

    def submit_count(self, batch_size, images):
        self._idx += batch_size

    def reset(self):
        self._idx = 0
        return True

    def _summarize_accuracy(self):
        return
