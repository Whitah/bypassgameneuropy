import numpy as np
import pygame
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import random

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
AGENT_SPEED_MS = 100  

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
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Генерация случайных препятствий
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < 0.1 and (i, j) != (0, 0) and (i, j) != (self.grid_size-1, self.grid_size-1):
                    self.grid[i, j] = 1
        
        self.agent_pos = [0, 0]
        self.target_pos = [self.grid_size - 1, self.grid_size - 1]
        self.grid[self.target_pos[0], self.target_pos[1]] = 2
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        
        return self.grid.copy(), {}
    
    def step(self, action):
        self.step_count += 1
        old_pos = self.agent_pos.copy()
        old_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
        
        if action == 0: self.agent_pos[0] -= 1
        elif action == 1: self.agent_pos[1] += 1
        elif action == 2: self.agent_pos[0] += 1
        elif action == 3: self.agent_pos[1] -= 1
        
        reward = -0.01
        terminated = False
        truncated = False
        
        if (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.grid_size or
            self.agent_pos[1] < 0 or self.agent_pos[1] >= self.grid_size or
            self.grid[self.agent_pos[0], self.agent_pos[1]] == 1):
            self.agent_pos = old_pos
            reward = -2.0
        else:
            new_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
            if new_distance < old_distance:
                progress = old_distance - new_distance
                reward = 1.0 + 0.5 * progress
            elif new_distance == old_distance:
                reward = -0.05
            else:
                reward = -0.2
            self.grid[old_pos[0], old_pos[1]] = 0
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        
        if self.agent_pos == self.target_pos:
            reward = 20.0
            terminated = True
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        return self.grid.copy(), reward, terminated, truncated, {}

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


def create_env_and_model(grid_size):
    env = MazeEnvWithLearning(grid_size)
    cell_size = min(50, 600 // grid_size)
    screen_width = grid_size * cell_size
    screen_height = grid_size * cell_size + 100
    try:
        model = PPO.load(f"ppo_maze_model_{grid_size}")
        message = f"Загружена модель для {grid_size}x{grid_size}!"
    except:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=20,
            gamma=0.98,
            ent_coef=0.01,
            clip_range=0.2,
            gae_lambda=0.95,
        )
        message = f"Создана новая модель для {grid_size}x{grid_size}"
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
                    
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing:
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    
                    if 0 <= row < current_grid_size and 0 <= col < current_grid_size and grid[row, col] not in [2, 3]:
                        grid[row, col] = draw_value
                        env.grid[row, col] = draw_value

        if current_time - last_move_time > AGENT_SPEED_MS and not paused:
            # Используем нейросеть для выбора действия
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            grid = obs.copy()
            agent_pos = env.agent_pos.copy()
            
            if terminated:
                message = "Нейросеть достигла цели!"
                # Обучение после достижения цели
                model.learn(total_timesteps=4000, reset_num_timesteps=False)
                training_episodes += 1
                model.save(f"ppo_maze_model_{current_grid_size}")
                # Новый эпизод
                obs, _ = env.reset()
                grid = obs.copy()
                agent_pos = env.agent_pos.copy()
                target_pos = env.target_pos.copy()
                message = f"Новый лабиринт! Обучено {training_episodes} раз"
            elif truncated:
                message = "Время вышло, обучение и перезапуск"
                # Обучение после таймаута
                model.learn(total_timesteps=2000, reset_num_timesteps=False)
                training_episodes += 1
                model.save(f"ppo_maze_model_{current_grid_size}")
                # Новый эпизод
                obs, _ = env.reset()
                grid = obs.copy()
                agent_pos = env.agent_pos.copy()
                target_pos = env.target_pos.copy()
                message = f"Новый лабиринт! Обучено {training_episodes} раз"
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

    pygame.quit()