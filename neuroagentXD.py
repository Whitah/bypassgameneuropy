import numpy as np
import pygame
import random
from collections import deque

# Начальные настройки
GRID_SIZE = 20
CELL_SIZE = 25
SCREEN_WIDTH = 800 # Фиксированная ширина для удобства меню
SCREEN_HEIGHT = 700 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
AGENT_SPEED_MS = 150

def get_next_best_step(grid, start_pos, target_pos, g_size):
    start = tuple(start_pos)
    goal = tuple(target_pos)
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return list(path[0]) if path else list(current)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = current[0] + dr, current[1] + dc
            if 0 <= r < g_size and 0 <= c < g_size:
                if grid[r, c] != 1 and (r, c) not in visited:
                    visited.add((r, c))
                    queue.append(((r, c), path + [(r, c)]))
    return None

def reset_game(g_size):
    new_grid = np.zeros((g_size, g_size), dtype=np.int8)
    a_pos = [0, 0]
    t_pos = [g_size - 1, g_size - 1]
    new_grid[t_pos[0], t_pos[1]] = 2
    new_grid[a_pos[0], a_pos[1]] = 3
    return new_grid, a_pos, t_pos

if __name__ == "__main__":
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Smart Agent: Pro Edition")
    font = pygame.font.SysFont('Arial', 22)
    small_font = pygame.font.SysFont('Arial', 16)
    
    # Игровые переменные
    current_grid_size = GRID_SIZE
    grid, agent_pos, target_pos = reset_game(current_grid_size)
    agent_color = BLUE
    
    message = "Нажми SPACE, чтобы запустить агента"
    last_move_time = pygame.time.get_ticks()
    
    running = True
    in_menu = False
    is_active = False # Движется ли шар
    is_drawing = False
    draw_value = 1

    clock = pygame.time.Clock()

    while running:
        current_time = pygame.time.get_ticks()
        # Динамический размер ячейки под размер поля
        CELL_SIZE = (SCREEN_HEIGHT - 150) // current_grid_size
        OFFSET_X = (SCREEN_WIDTH - (current_grid_size * CELL_SIZE)) // 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    in_menu = not in_menu
                if event.key == pygame.K_SPACE:
                    is_active = not is_active
                    message = "Поехали!" if is_active else "Пауза. Нажми SPACE"
                if event.key == pygame.K_TAB: # Генерация лабиринта
                    grid, agent_pos, target_pos = reset_game(current_grid_size)
                    for r in range(current_grid_size):
                        for c in range(current_grid_size):
                            if random.random() > 0.7 and grid[r, c] == 0:
                                grid[r, c] = 1
                    message = "Лабиринт создан!"
                if event.key in [pygame.K_LCTRL, pygame.K_RCTRL]: # Очистка
                    grid, agent_pos, target_pos = reset_game(current_grid_size)
                    message = "Поле очищено."

                # Управление в меню
                if in_menu:
                    if event.key == pygame.K_1: current_grid_size = 10; grid, agent_pos, target_pos = reset_game(10)
                    if event.key == pygame.K_2: current_grid_size = 20; grid, agent_pos, target_pos = reset_game(20)
                    if event.key == pygame.K_3: current_grid_size = 30; grid, agent_pos, target_pos = reset_game(30)
                    if event.key == pygame.K_r: agent_color = RED
                    if event.key == pygame.K_b: agent_color = BLUE
                    if event.key == pygame.K_p: agent_color = PURPLE

            # Рисование (только если не в меню)
            if not in_menu:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        is_drawing = True
                        mx, my = event.pos
                        col, row = (mx - OFFSET_X) // CELL_SIZE, my // CELL_SIZE
                        if 0 <= row < current_grid_size and 0 <= col < current_grid_size:
                            if grid[row, col] not in [2, 3]:
                                draw_value = 0 if grid[row, col] == 1 else 1
                                grid[row, col] = draw_value
                elif event.type == pygame.MOUSEBUTTONUP:
                    is_drawing = False
                elif event.type == pygame.MOUSEMOTION and is_drawing:
                    mx, my = event.pos
                    col, row = (mx - OFFSET_X) // CELL_SIZE, my // CELL_SIZE
                    if 0 <= row < current_grid_size and 0 <= col < current_grid_size:
                        if grid[row, col] not in [2, 3]:
                            grid[row, col] = draw_value

        # Логика движения агента
        if is_active and not in_menu:
            if current_time - last_move_time > AGENT_SPEED_MS:
                if agent_pos == target_pos:
                    message = "Победа! Очистка..."
                    is_active = False
                else:
                    next_step = get_next_best_step(grid, agent_pos, target_pos, current_grid_size)
                    if next_step:
                        grid[agent_pos[0], agent_pos[1]] = 0
                        agent_pos = next_step
                        grid[agent_pos[0], agent_pos[1]] = 3
                    else:
                        message = "Я застрял! Убери преграду."
                last_move_time = current_time

        # Отрисовка
        window.fill(WHITE)
        
        # Сетка и поле
        for r in range(current_grid_size):
            for c in range(current_grid_size):
                rect = pygame.Rect(OFFSET_X + c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(window, GRAY, rect, 1)
                val = grid[r, c]
                if val == 1: pygame.draw.rect(window, BLACK, rect)
                elif val == 2: pygame.draw.rect(window, GREEN, rect)
                elif val == 3: pygame.draw.circle(window, agent_color, rect.center, CELL_SIZE // 2 - 2)

        # Панель управления (снизу)
        panel_rect = pygame.Rect(0, SCREEN_HEIGHT - 150, SCREEN_WIDTH, 150)
        pygame.draw.rect(window, (240, 240, 240), panel_rect)
        pygame.draw.line(window, BLACK, (0, SCREEN_HEIGHT - 150), (SCREEN_WIDTH, SCREEN_HEIGHT - 150), 2)
        
        # Сообщения
        msg_surf = font.render(message, True, BLACK)
        window.blit(msg_surf, (20, SCREEN_HEIGHT - 130))
        
        # Подсказки (еле видны)
        hints = "Space: Старт/Пауза | Tab: Лабиринт | Ctrl: Очистить | Esc: Настройки"
        hint_surf = small_font.render(hints, True, (150, 150, 150))
        window.blit(hint_surf, (20, SCREEN_HEIGHT - 40))

        # Отрисовка меню поверх всего
        if in_menu:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 200))
            window.blit(overlay, (0, 0))
            
            menu_text = [
                "--- НАСТРОЙКИ (Esc для выхода) ---",
                "Размер поля: 1: 10 | 2: 20 | 3: 30",
                "Цвет шарика: R: Красный | B: Синий | P: Фиолетовый",
                "",
                f"Текущий размер: {current_grid_size}x{current_grid_size}"
            ]
            for i, line in enumerate(menu_text):
                t = font.render(line, True, BLACK)
                window.blit(t, (SCREEN_WIDTH // 2 - 200, 150 + i * 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()