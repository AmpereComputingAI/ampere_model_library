import random

from utils.helpers import DatasetStub


class StableDiffusion(DatasetStub):

    def __init__(self):
        self._idx = 0
        self.available_instances = 100000

    def get_input(self):
        adjectives = ['professional', 'enormous', 'fun', 'beautiful', 'small', 'ugly']
        nouns = ['dog', 'cat', 'astronaut', 'person', 'knight', 'horse', 'soldier']
        actions = ['runs', 'jumps', 'rides a triceratops', 'rides a bike', 'eats a burger', 'washes clothes']
        adverbs = ['quickly', 'slowly', 'loudly']

        return f'{random.choice(adjectives)} {random.choice(nouns)} {random.choice(actions)} {random.choice(adverbs)}'

    def submit_count(self):
        self._idx += 1

    def reset(self):
        self._idx = 0
        return True

    def summarize_accuracy(self):
        print("accuracy metrics for this model are under development")
        return {}
