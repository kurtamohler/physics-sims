import pygame

class Sim:
    def __init__(self):
        raise NotImplementedError('method must be defined by subclass')

    def update(self, sim_runner, dt):
        raise NotImplementedError('method must be defined by subclass')

    def state(self):
        raise NotImplementedError('method must be defined by subclass')
    
    def draw(self, sim_runner):
        raise NotImplementedError('method must be defined by subclass')

    def handle_event(self, event: pygame.event.Event):
        pass