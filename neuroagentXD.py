import numpy as np
import pygame
from collections import deque

GRID_SIZE = 20    
CELL_SIZE = 30    
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 100 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)     
BLUE = (0, 0, 255)    
GREEN = (0, 255, 0)   
GRAY = (200, 200, 200)
RED = (255, 0, 0)    
AGENT_SPEED_MS = 200  

def get_next_best_step(grid, start_pos, target_pos):
    """
    Поиск пути в ширину (BFS). Строит оптимальный маршрут к цели.
    """
    start = tuple(start_pos)
    goal = tuple(target_pos)
    
    queue = deque([(start, [])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal:
            return list(path[0]) if path else list(current)
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = current[0] + dr, current[1] + dc
            
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                if grid[r, c] != 1 and (r, c) not in visited:
                    visited.add((r, c))
                    queue.append(((r, c), path + [(r, c)]))
                    
    return None


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Умный Агент: Большое поле")
    font = pygame.font.SysFont('Arial', 24)
    clock = pygame.time.Clock()

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    agent_pos = [0, 0]
    target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
    
    grid[target_pos[0], target_pos[1]] = 2
    grid[agent_pos[0], agent_pos[1]] = 3
    
    message = "Поле чистое. Рисуй стены!"
    
    last_move_time = pygame.time.get_ticks()
    running = True
    
 
    is_drawing = False
    draw_value = 1 # 1 - рисуем стену, 0 - стираем стену

    while running:
        current_time = pygame.time.get_ticks()
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Лкм
                    is_drawing = True
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    
                    if 0 <= row < GRID_SIZE and grid[row, col] not in [2, 3]:
                        # Если кликнули по пустому - рисуем (1), если по стене - стираем (0)
                        draw_value = 0 if grid[row, col] == 1 else 1
                        grid[row, col] = draw_value
            
            # отпустили мышь
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_drawing = False
                    
            # зажать лкм рисует путь
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing:
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    
                    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and grid[row, col] not in [2, 3]:
                        grid[row, col] = draw_value

        # логика
        if current_time - last_move_time > AGENT_SPEED_MS:
            if agent_pos == target_pos:
                message = "Цель достигнута!"
                # последний кадр победы перед паузой
                window.fill(WHITE)
                for row in range(GRID_SIZE):
                    for col in range(GRID_SIZE):
                        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(window, GRAY, rect, 1)
                        if grid[row, col] == 1: pygame.draw.rect(window, BLACK, rect)
                        elif grid[row, col] == 2: pygame.draw.rect(window, GREEN, rect)
                        elif grid[row, col] == 3: pygame.draw.circle(window, BLUE, rect.center, CELL_SIZE // 2 - 3)
                pygame.draw.rect(window, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
                window.blit(font.render(message, True, BLACK), (10, SCREEN_HEIGHT - 70))
                pygame.display.flip()
                
                pygame.time.wait(1500) # Пауза перед рестартом
                
                # удаление препятствий 
                grid.fill(0)
                
                # начало заново
                agent_pos = [0, 0]
                target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
                grid[target_pos[0], target_pos[1]] = 2
                grid[agent_pos[0], agent_pos[1]] = 3
                
                message = "Погнали! Строй мне лабиринт."
            else:
                next_step = get_next_best_step(grid, agent_pos, target_pos)
                
                if next_step:
                    grid[agent_pos[0], agent_pos[1]] = 0
                    agent_pos = next_step
                    grid[agent_pos[0], agent_pos[1]] = 3
                    
                    if message == "Путь заблокирован! Я застрял!":
                        message = "Нашел обход! Иду дальше."
                else:
                    message = "Путь заблокирован! Я застрял!"
            
            last_move_time = current_time

        # графика
        window.fill(WHITE)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(window, GRAY, rect, 1)
                
                cell_val = grid[row, col]
                if cell_val == 1:   
                    pygame.draw.rect(window, BLACK, rect)
                elif cell_val == 2: 
                    pygame.draw.rect(window, GREEN, rect)
                elif cell_val == 3: 
                    color = RED if message == "Путь заблокирован!!!" else BLUE
                    pygame.draw.circle(window, color, rect.center, CELL_SIZE // 2 - 3)

        pygame.draw.rect(window, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        text_surface = font.render(message, True, BLACK)
        window.blit(text_surface, (10, SCREEN_HEIGHT - 70))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()