import pygame
import os


class Buzzer:

    def __init__(self):

        pygame.mixer.init()

        self.sound = pygame.mixer.Sound(
            os.path.join("assets", "warning.wav")
        )

        self.playing = False

    def on(self):

        if not self.playing:

            self.sound.play(-1)      # Loop continuously

            self.playing = True

    def off(self):

        if self.playing:

            self.sound.stop()

            self.playing = False