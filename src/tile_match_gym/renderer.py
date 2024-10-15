import colorsys
from typing import Optional, Union

import numpy as np
import pygame


class Renderer:
    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            num_colours: int,
            num_moves: int,
            window_size: int = 512,
            render_mode: Optional[str] = "human"
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_colours = num_colours
        self.num_moves = num_moves
        self.window_size = window_size
        self.render_mode = render_mode

        self.screen = None
        self.width = None
        self.height = None

        self.colour_map = []
        for i in range(1, num_colours + 1):  # Skip white
            # Evenly space the hue value            hue = i / num_colours
            # Fix saturation and lightness at 0.6 and 0.5 for vivid colors
            saturation = 0.6
            lightness = 0.5
            hue = i / num_colours
            # Convert the HSL values to RGB (returns values in range [0, 1], so scale to [0, 255])
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            rgb = tuple(int(val * 255) for val in rgb)  # Scale to [0, 255]
            self.colour_map.append(rgb)

    def render(self, board: np.ndarray, num_moves_left: int) -> Union[None, np.ndarray]:
        if self.screen is None:
            self._init_pygame()

        white = (255, 255, 255)
        black = (0, 0, 0)
        self.screen.fill(white)

        board_width = self.num_cols * (self.tile_size + self.spacing) - self.spacing
        board_height = self.num_rows * (self.tile_size + self.spacing) - self.spacing
        board_x = (self.screen_width - board_width) // 2
        board_y = self.text_area_height + (self.screen_height - self.text_area_height - board_height) // 2

        # Draw the board
        for row in range(self.num_rows):
            for col in range(self.num_cols):

                tile_colour = board[0, row, col]
                tile_type = board[1, row, col]

                if tile_colour == 0:
                    color = black
                else:
                    color = self.colour_map[tile_colour - 1]

                x = board_x + col * (self.tile_size + self.spacing)
                y = board_y + row * (self.tile_size + self.spacing)


                if tile_type == 1:  # Ordinary tile
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))
                elif tile_type == 2:  # Vertical laser
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))

                    pygame.draw.rect(self.screen, black,(x + self.tile_size // 3, y, self.tile_size // 3, self.tile_size))
                elif tile_type == 3:  # Horizontal laser
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))

                    pygame.draw.rect(self.screen, black,(x, y + self.tile_size // 3, self.tile_size, self.tile_size // 3))
                elif tile_type == 4:  # Bomb
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))
                    pygame.draw.circle(self.screen, black, (x + self.tile_size // 2, y + self.tile_size // 2), self.tile_size // 3)
                elif tile_type == -1:  # Cookie
                    pygame.draw.polygon(self.screen, black, [
                        (x + self.tile_size // 2, y),
                        (x + self.tile_size, y + self.tile_size // 2),
                        (x + self.tile_size // 2, y + self.tile_size),
                        (x, y + self.tile_size // 2)
                    ])

        # Display moves left at the top in the center
        font_size = int(self.text_area_height * 0.8)
        font = pygame.font.SysFont("helvetica", font_size)
        text_surface = font.render(f"Moves Left: {num_moves_left}", True, black)
        text_x = (self.screen_width - text_surface.get_width()) // 2
        text_y = ((self.text_area_height - text_surface.get_height() + self.tile_size) // 2)

        self.screen.blit(text_surface, (text_x, text_y))
        pygame.display.flip()

        if self.render_mode == "rgb_array":
            img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            return img


    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Tile Match")
        info = pygame.display.Info()

        board_screen_ratio = 0.3
        screen_width, screen_height = info.current_w * board_screen_ratio, info.current_h * board_screen_ratio
        self.screen_width = screen_width * (self.num_cols / self.num_rows)
        self.screen_height = screen_height * (self.num_rows / self.num_cols)

        text_area_height_ratio = 0.15
        margin_ratio = 0.05
        spacing_ratio = 0.05
        self.text_area_height = min(25, self.screen_height * text_area_height_ratio)

        # Available height and width for the board after removing margins and text area
        board_render_width = self.screen_width * (1 - 2 * margin_ratio)
        board_render_height = (self.screen_height - self.text_area_height) * (1 - 2 * margin_ratio)

        max_tile_width = board_render_width / (self.num_cols + (self.num_cols - 1) * spacing_ratio)
        max_tile_height = board_render_height / (self.num_rows + (self.num_rows - 1) * spacing_ratio)
        self.tile_size = min(int(max_tile_width), int(max_tile_height))
        self.spacing = self.tile_size * spacing_ratio

        # Calculate the actual board width and height (including tiles and spacings)
        self.board_render_width = self.num_cols * self.tile_size + (self.num_cols - 1) * self.spacing
        self.board_render_height = self.num_rows * self.tile_size + (self.num_rows - 1) * self.spacing

        self.screen_height = self.board_render_height + self.text_area_height + 2 * self.tile_size
        self.screen_width = self.board_render_width + 2 * self.tile_size

        font_size = int(self.text_area_height * 0.8)
        font = pygame.font.SysFont("helvetica", font_size)
        max_text = font.render(f"Moves Left: {self.num_moves}", True, (0, 0, 0))
        min_width = max_text.get_width() + 2 * self.tile_size

        self.screen_width = max(self.screen_width, min_width)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))


    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
