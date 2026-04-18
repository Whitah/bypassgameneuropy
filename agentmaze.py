import numpy as np
import pygame
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random
import os

GRID_SIZE = 30    
CELL_SIZE = 20    
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 100 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)     
BLUE = (0, 0, 255)    
GREEN = (0, 255, 0)   
GRAY = (200, 200, 200)
RED = (255, 0, 0)    
AGENT_SPEED_MS = 100  # Увеличено для замедления агента  

class MazeEnvWithLearning(gym.Env):
    """Среда лабиринта с обучением PPO"""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0=вверх, 1=вправо, 2=вниз, 3=влево
        self.observation_space = spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int8)
        self.max_steps = 500
        self.step_count = 0
        self.episode_count = 0
        self.visited = set()
        self.next_grid = None
        self.last_action = None
        self.last_collision = False
        self.last_pos = None
        self.position_history = deque(maxlen=12)
        self.blocked_dirs = {}
        
    def generate_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < 0.1 and (i, j) != (0, 0) and (i, j) != (self.grid_size-1, self.grid_size-1):
                    grid[i, j] = 1
        return grid
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        
        regenerate = options.get('regenerate', True) if options else True
        restart_current = options.get('restart_current', False) if options else False
        
        if restart_current:
            # Перезапуск текущего лабиринта: сбрасываем позицию и visited, но сохраняем grid
            # Очищаем старую позицию агента
            self.grid[self.grid == 3] = 0
            self.agent_pos = [0, 0]
            self.target_pos = [self.grid_size - 1, self.grid_size - 1]
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
            self.visited = set([tuple(self.agent_pos)])
            return self.grid.copy(), {}
        elif self.next_grid is not None and regenerate:
            self.grid = self.next_grid
        else:
            self.grid = self.generate_grid()
        
        if regenerate and not restart_current:
            self.next_grid = self.generate_grid()
        
        self.agent_pos = [0, 0]
        self.target_pos = [self.grid_size - 1, self.grid_size - 1]
        self.grid[self.target_pos[0], self.target_pos[1]] = 2
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        self.visited = set([tuple(self.agent_pos)])
        self.last_action = None
        self.last_collision = False
        self.last_pos = None
        self.position_history = deque(maxlen=12)
        self.position_history.append(tuple(self.agent_pos))
        self.blocked_dirs = {}
        
        return self.grid.copy(), {}
    
    def step(self, action):
        self.step_count += 1
        old_pos = self.agent_pos.copy()
        old_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
        
        if self.last_collision and self.last_pos == tuple(old_pos) and action == self.last_action:
            # Не повторяем немедленное движение в ту же стену
            for alt_action in [0, 1, 2, 3]:
                if alt_action == action:
                    continue
                alt_pos = old_pos.copy()
                if alt_action == 0:
                    alt_pos[0] -= 1
                elif alt_action == 1:
                    alt_pos[1] += 1
                elif alt_action == 2:
                    alt_pos[0] += 1
                elif alt_action == 3:
                    alt_pos[1] -= 1
                if 0 <= alt_pos[0] < self.grid_size and 0 <= alt_pos[1] < self.grid_size and self.grid[alt_pos[0], alt_pos[1]] != 1:
                    action = alt_action
                    break
        self.agent_pos = old_pos.copy()
        if action == 0:
            self.agent_pos[0] -= 1
        elif action == 1:
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[0] += 1
        elif action == 3:
            self.agent_pos[1] -= 1

        reward = -0.1
        terminated = False
        truncated = False
        
        # Штраф за приближение к границам и стенам (lookahead)
        border_penalty = 0
        if old_pos[0] <= 1 or old_pos[0] >= self.grid_size - 2:
            border_penalty -= 0.15
        if old_pos[1] <= 1 or old_pos[1] >= self.grid_size - 2:
            border_penalty -= 0.15

        collision = (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.grid_size or
                     self.agent_pos[1] < 0 or self.agent_pos[1] >= self.grid_size or
                     self.grid[self.agent_pos[0], self.agent_pos[1]] == 1)

        horizontal_bonus = False
        if collision:
            # Агент получил информацию о блоке и ищет соседнюю клетку, чтобы не останавливаться
            is_boundary_collision = (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.grid_size or
                                      self.agent_pos[1] < 0 or self.agent_pos[1] >= self.grid_size)
            reward = -3.0 if is_boundary_collision else -2.5
            self.agent_pos = old_pos
            free_neighbors = []
            visited_neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = old_pos[0] + dr, old_pos[1] + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.grid[nr, nc] != 1:
                    if (nr, nc) not in self.visited:
                        free_neighbors.append((nr, nc))
                    else:
                        visited_neighbors.append((nr, nc))
            chosen = None
            if free_neighbors:
                fresh_free = [pos for pos in free_neighbors if pos not in self.position_history]
                if fresh_free:
                    free_neighbors = fresh_free
                chosen = min(
                    free_neighbors,
                    key=lambda pos: abs(pos[0] - self.target_pos[0]) + abs(pos[1] - self.target_pos[1])
                )
                self.agent_pos = list(chosen)
                new_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
                if new_distance < old_distance:
                    progress = old_distance - new_distance
                    reward = 1.0 + progress
                else:
                    reward = -1.5
                if self.agent_pos[0] == old_pos[0] and tuple(self.agent_pos) not in self.position_history:
                    horizontal_bonus = True
                # Штраф за оставание близко к границе
                if self.agent_pos[0] <= 1 or self.agent_pos[0] >= self.grid_size - 2:
                    reward -= 0.2
                if self.agent_pos[1] <= 1 or self.agent_pos[1] >= self.grid_size - 2:
                    reward -= 0.2
            elif visited_neighbors:
                fresh_visited = [pos for pos in visited_neighbors if pos not in self.position_history]
                if fresh_visited:
                    visited_neighbors = fresh_visited
                def neighbor_score(pos):
                    unvisited_count = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ar, ac = pos[0] + dr, pos[1] + dc
                        if 0 <= ar < self.grid_size and 0 <= ac < self.grid_size:
                            if self.grid[ar, ac] == 0 and (ar, ac) not in self.visited:
                                unvisited_count += 1
                    return (-unvisited_count, abs(pos[0] - self.target_pos[0]) + abs(pos[1] - self.target_pos[1]))
                chosen = min(visited_neighbors, key=neighbor_score)
                self.agent_pos = list(chosen)
                reward = -1.0
                if self.agent_pos[0] == old_pos[0] and tuple(self.agent_pos) not in self.position_history:
                    horizontal_bonus = True
            if chosen is not None:
                self.grid[old_pos[0], old_pos[1]] = 0
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
                self.visited.add(tuple(self.agent_pos))
                self.position_history.append(tuple(self.agent_pos))
                if horizontal_bonus:
                    reward += 0.5
            self.last_collision = True
            self.last_action = action
            self.last_pos = tuple(old_pos)
        else:
            self.last_collision = False
            self.last_action = action
            self.last_pos = tuple(old_pos)
            if tuple(self.agent_pos) in self.visited:
                # Штраф за повторение позиции, но используем историю, чтобы избегать циклов
                reward = -1.2
                self.grid[old_pos[0], old_pos[1]] = 0
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
                self.visited.add(tuple(self.agent_pos))
                self.position_history.append(tuple(self.agent_pos))
                # Штраф за остановку близко к границе
                if self.agent_pos[0] <= 1 or self.agent_pos[0] >= self.grid_size - 2:
                    reward -= 0.1
                if self.agent_pos[1] <= 1 or self.agent_pos[1] >= self.grid_size - 2:
                    reward -= 0.1
            else:
                new_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
                if new_distance < old_distance:
                    progress = old_distance - new_distance
                    reward = 2.0 + 1.5 * progress  # Повышенная награда за прогресс
                elif new_distance == old_distance:
                    reward = -0.3
                else:
                    reward = -1.2
                if self.agent_pos[0] == old_pos[0] and tuple(self.agent_pos) not in self.position_history:
                    horizontal_bonus = True
                # Штраф за движение к границе
                if self.agent_pos[0] <= 1 or self.agent_pos[0] >= self.grid_size - 2:
                    reward -= 0.1
                if self.agent_pos[1] <= 1 or self.agent_pos[1] >= self.grid_size - 2:
                    reward -= 0.1
                self.grid[old_pos[0], old_pos[1]] = 0
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
                self.visited.add(tuple(self.agent_pos))
                self.position_history.append(tuple(self.agent_pos))
        
        if self.agent_pos == self.target_pos:
            reward = 50.0  # Значительно повышена награда за достижение цели
            terminated = True
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        reward += border_penalty  # Добавляем штраф за приближение к границам
        return self.grid.copy(), reward, terminated, truncated, {}

def get_next_best_step(grid, start_pos, target_pos, visited=None):
    """
    Поиск пути в ширину (BFS). Строит оптимальный маршрут к цели, избегая посещенных позиций.
    """
    if visited is None:
        visited_bfs = set()
    else:
        visited_bfs = visited.copy()
    
    grid_size = grid.shape[0]
    start = tuple(start_pos)
    goal = tuple(target_pos)
    
    queue = deque([(start, [])])
    visited_bfs.add(start)
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal:
            return list(path[0]) if path else list(current)
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = current[0] + dr, current[1] + dc
            
            if 0 <= r < grid_size and 0 <= c < grid_size:
                if grid[r, c] != 1 and (r, c) not in visited_bfs:
                    visited_bfs.add((r, c))
                    queue.append(((r, c), path + [(r, c)]))
                    
    return None


def create_env_and_model(grid_size):
    # Создаем папку для сохранения моделей, если её нет
    os.makedirs("models", exist_ok=True)
    
    env = MazeEnvWithLearning(grid_size)
    cell_size = min(50, 600 // grid_size)
    screen_width = grid_size * cell_size
    screen_height = grid_size * cell_size + 100
    
    model_path = f"models/ppo_maze_model_{grid_size}"
    
    try:
        # Пытаемся загрузить существующую модель
        model = PPO.load(model_path, env=env)
        message = f"Загружена модель для {grid_size}x{grid_size}! Продолжаем обучение..."
    except (FileNotFoundError, Exception) as e:
        # Если модель не найдена или ошибка загрузки, создаем новую
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=8e-4,  # Увеличена скорость обучения для быстрого обучения
            n_steps=2048,  # Уменьшено для более частого обновления
            batch_size=64,  # Уменьшено для более частого обновления
            n_epochs=30,  # Увеличено количество эпох
            gamma=0.99,  # Увеличено для лучшей оценки будущих вознаграждений
            ent_coef=0.02,  # Увеличено для лучшего исследования
            clip_range=0.3,  # Увеличено для большего шага обновления
            gae_lambda=0.97,  # Увеличено для стабильности
        )
        message = f"Создана новая модель для {grid_size}x{grid_size} с оптимизированными параметрами"
    return env, model, cell_size, screen_width, screen_height, message


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    
    current_grid_size = 30
    env, model, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, message = create_env_and_model(current_grid_size)
    
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Умный Агент: Большое поле с Нейросетью")
    font = pygame.font.SysFont('Arial', 18)
    clock = pygame.time.Clock()
    
    obs, _ = env.reset()    
    grid = obs.copy()
    agent_pos = env.agent_pos.copy()
    target_pos = env.target_pos.copy()
    
    last_move_time = pygame.time.get_ticks()
    running = True
    training_episodes = 0
    paused = False
    
    is_drawing = False
    draw_value = 1
    last_grid_state = grid.copy()  # Отслеживаем последнее состояние сетки
    grid_changed = False

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    message = "Пауза: SPACE - продолжить, R - новый лабиринт, Q - выход, 1-5 - размер" if paused else "Продолжено"
                elif event.key == pygame.K_r and paused:
                    # Перегенерировать лабиринт
                    obs, _ = env.reset()
                    grid = obs.copy()
                    agent_pos = env.agent_pos.copy()
                    target_pos = env.target_pos.copy()
                    message = "Новый лабиринт сгенерирован"
                elif event.key == pygame.K_q:
                    running = False
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5] and paused:
                    size_map = {pygame.K_1: 10, pygame.K_2: 20, pygame.K_3: 30, pygame.K_4: 40, pygame.K_5: 50}
                    new_size = size_map[event.key]
                    if new_size != current_grid_size:
                        current_grid_size = new_size
                        env, model, CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, message = create_env_and_model(current_grid_size)
                        window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                        obs, _ = env.reset()
                        grid = obs.copy()
                        agent_pos = env.agent_pos.copy()
                        target_pos = env.target_pos.copy()
                        training_episodes = 0
                        message = f"Размер изменен на {current_grid_size}x{current_grid_size}"
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_drawing = True
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    
                    if 0 <= row < current_grid_size and grid[row, col] not in [2, 3]:
                        draw_value = 0 if grid[row, col] == 1 else 1
                        grid[row, col] = draw_value
                        env.grid[row, col] = draw_value
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_drawing = False
                    grid_changed = True  # Отмечаем, что лабиринт был изменен
                    
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing:
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    
                    if 0 <= row < current_grid_size and 0 <= col < current_grid_size and grid[row, col] not in [2, 3]:
                        grid[row, col] = draw_value
                        env.grid[row, col] = draw_value

        if current_time - last_move_time > AGENT_SPEED_MS and not paused:
            # Пытаемся быстро адаптироваться, если лабиринт был изменен пользователем
            if grid_changed:
                # Быстрая микрообучение для адаптации к новому лабиринту
                model.learn(total_timesteps=256, reset_num_timesteps=False)  # Минимальное быстрое обучение
                grid_changed = False
            
            # Используем нейросеть для выбора действия (детерминированно для скорости)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            grid = obs.copy()
            agent_pos = env.agent_pos.copy()
            
            if terminated:
                message = "Нейросеть достигла цели! Обучение..."
                # Интенсивное обучение после достижения цели
                model.learn(total_timesteps=8000, reset_num_timesteps=False)
                training_episodes += 1
                os.makedirs("models", exist_ok=True)
                model.save(f"models/ppo_maze_model_{current_grid_size}")
                # Новый эпизод
                obs, _ = env.reset()
                grid = obs.copy()
                agent_pos = env.agent_pos.copy()
                target_pos = env.target_pos.copy()
                grid_changed = False  # Сбрасываем флаг при новом лабиринте
                message = f"Новый лабиринт! Усиленных обучений: {training_episodes}"
            elif truncated:
                message = "Время вышло, перезапуск текущего лабиринта"
                # Перезапуск текущего лабиринта без дополнительного обучения
                obs, _ = env.reset(options={'restart_current': True})
                grid = obs.copy()
                agent_pos = env.agent_pos.copy()
                target_pos = env.target_pos.copy()
                grid_changed = False  # Сбрасываем флаг при новом лабиринте
                message = f"Текущий лабиринт перезапущен! Обучено {training_episodes} раз"
            else:
                message = f"Шаг {env.step_count}, награда: {reward:.2f}"
            
            last_move_time = current_time

        # Графика
        window.fill(WHITE)

        for row in range(current_grid_size):
            for col in range(current_grid_size):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(window, GRAY, rect, 1)
                
                cell_val = grid[row, col]
                if cell_val == 1:   
                    pygame.draw.rect(window, BLACK, rect)
                elif cell_val == 2: 
                    pygame.draw.rect(window, GREEN, rect)
                elif cell_val == 3: 
                    pygame.draw.circle(window, BLUE, rect.center, CELL_SIZE // 2 - 2)

        pygame.draw.rect(window, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        text_surface = font.render(message, True, BLACK)
        window.blit(text_surface, (10, SCREEN_HEIGHT - 70))
        
        if paused:
            pause_text = font.render("ПАУЗА", True, RED)
            window.blit(pause_text, (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 70))

        pygame.display.flip()
        clock.tick(60)

    # Сохраняем модель перед выходом
    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_maze_model_{current_grid_size}")
    print(f"Модель сохранена в models/ppo_maze_model_{current_grid_size}")
    
    pygame.quit()
