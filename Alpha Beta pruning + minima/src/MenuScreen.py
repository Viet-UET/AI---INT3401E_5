import sys

import pygame
import pygame_menu

from src.config import *


class MenuScreen:
    def __init__(self, screen):
        self.screen = screen
        # Setting menu background
        self.bg = pygame.transform.scale(pygame.image.load(".\data\images\gdh.png"), (WIDTH_WINDOW, HEIGHT_WINDOW))
        # theme setting
        font = pygame_menu.font.FONT_8BIT

        my_theme_harmonious = pygame_menu.Theme(
            background_color=(28, 30, 34, 180),      # Xám than đậm, hơi trong suốt
            widget_background_color=(50, 54, 63, 220), # Xám xanh nhạt hơn, ít trong suốt hơn
            widget_font_color=(210, 210, 215),     # Trắng ngà/Xám rất nhạt
            widget_margin=(0, 10),
            widget_padding=10,
            widget_font=font,
            widget_font_size=24,
            title_font_size=24
        )

        # Sử dụng theme mới này khi tạo menu
        self.menu = pygame_menu.Menu('Chess game', 300, 300,
                                     theme=my_theme_harmonious) # Thay đổi ở đây

        

        self.menu.add.button("Play", button_id='PvP')
        self.menu.add.button("Play with AI", button_id='PvC')
        self.menu.add.button("Comp vs Comp", button_id='CvC')
        self.menu.add.button("Quit", sys.exit, 1)

        self.running = True

    def mainLoop(self):
        while self.running:
            events = pygame.event.get()
            self.menu.update(events)

            self._draw_background()
            self.menu.draw(self.screen)
            pygame.display.update()

    def _draw_background(self):
        self.screen.blit(self.bg, (0, 0))
