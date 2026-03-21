import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. НАСТРОЙКИ СРЕДЫ ---
GRID_SIZE = 10
CELL_SIZE = 50
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 100
MAX_STEPS = 200
MODEL_PATH = "ppo_maze"

WHITE, BLACK, BLUE, GREEN, GRAY = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0), (200, 200, 200)

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Вывод прогресса каждые 2000 шагов, чтобы терминал не "молчал"
        if self.num_timesteps % 2000 == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            print(f"🛠 Этап обучения: {self.num_timesteps} / {self.total_timesteps} шагов ({progress:.0f}%)")
        return True

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(GRID_SIZE, GRID_SIZE, 1), dtype=np.uint8)
        self.window = None
        self.clock = None
        self.message = "Инициализация..."

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = [0, 0]
        self.target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        
        self.walls = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.85, 0.15])
        self.walls[self.agent_pos[0], self.agent_pos[1]] = 0
        self.walls[self.target_pos[0], self.target_pos[1]] = 0
        
        self.message = "Новая попытка..."
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.uint8)
        obs[self.walls == 1] = 85
        obs[self.target_pos[0], self.target_pos[1], 0] = 170
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 255
        return obs

    def _get_distance(self):
        return abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])

    def step(self, action):
        self.current_step += 1
        old_dist = self._get_distance()
        
        new_pos = list(self.agent_pos)
        if action == 0: new_pos[0] -= 1
        elif action == 1: new_pos[1] += 1
        elif action == 2: new_pos[0] += 1
        elif action == 3: new_pos[1] -= 1

        reward = -0.1 
        terminated = False
        truncated = False
        
        if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and self.walls[new_pos[0], new_pos[1]] == 0:
            self.agent_pos = new_pos
            new_dist = self._get_distance()
            reward += (old_dist - new_dist) * 0.5 
            self.message = "Двигаюсь..."
        else:
            reward = -1.0
            self.message = "Стена!"

        if self.agent_pos == self.target_pos:
            reward = 10.0
            terminated = True
            self.message = "Цель достигнута!"
        
        if self.current_step >= MAX_STEPS:
            truncated = True
            self.message = "Превышен лимит шагов."

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Нейро-Агент (CnnPolicy)")
            self.font = pygame.font.SysFont('Arial', 24)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // CELL_SIZE, y // CELL_SIZE
                if row < GRID_SIZE and [row, col] not in [self.agent_pos, self.target_pos]:
                    self.walls[row, col] = 1 - self.walls[row, col]

        self.window.fill(WHITE)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.window, GRAY, rect, 1)
                
                if self.walls[row, col] == 1:
                    pygame.draw.rect(self.window, BLACK, rect)
                elif [row, col] == self.target_pos:
                    pygame.draw.rect(self.window, GREEN, rect)
                elif [row, col] == self.agent_pos:
                    pygame.draw.circle(self.window, BLUE, rect.center, CELL_SIZE // 2 - 5)

        pygame.draw.rect(self.window, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        text_surface = self.font.render(self.message, True, BLACK)
        self.window.blit(text_surface, (10, SCREEN_HEIGHT - 70))

        pygame.display.flip()
        self.clock.tick(15)

if __name__ == "__main__":
    env = MazeEnv()
    
    # Проверяем, есть ли уже обученная модель
    if os.path.exists(MODEL_PATH + ".zip"):
        print("📁 Найдена сохраненная модель! Загружаю...")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("🚀 Сохраненной модели нет. Начинаю обучение (CnnPolicy) на CPU...")
        total_steps = 30000 # Снижено до 30к для быстрого первого теста (займет пару минут)
        
        # verbose=1 включает встроенный подробный вывод логов (FPS, loss и т.д.)
        # device="cpu" предотвращает зависания при поиске видеокарты
        model = PPO("CnnPolicy", env, verbose=1, device="cpu")
        callback = ProgressCallback(total_steps)
        
        model.learn(total_timesteps=total_steps, callback=callback)
        model.save(MODEL_PATH)
        print("✅ Обучение завершено и модель сохранена! Запускаю симуляцию.")

    obs, info = env.reset()
    
    while True:
        env.render()
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.1) 
        
        if terminated or truncated:
            env.render()
            time.sleep(1)
            obs, info = env.reset()