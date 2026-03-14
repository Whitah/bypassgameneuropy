import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from stable_baselines3 import PPO

# --- 1. НАСТРОЙКИ СРЕДЫ ---
GRID_SIZE = 10
CELL_SIZE = 50
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 100 # +100 пикселей для панели текста

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)    # Агент
GREEN = (0, 255, 0)   # Цель
GRAY = (200, 200, 200)

class MazeEnv(gym.Env):
    """Кастомная среда лабиринта для обучения нейросети"""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(MazeEnv, self).__init__()
        
        # Действия: 0: Вверх, 1: Вправо, 2: Вниз, 3: Влево
        self.action_space = spaces.Discrete(4)
        
        # Наблюдение: матрица поля. 0 - пусто, 1 - стена, 2 - цель, 3 - агент
        self.observation_space = spaces.Box(low=0, high=3, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        self.window = None
        self.clock = None
        self.message = "Инициализация..."

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        # Стартовые позиции
        self.agent_pos = [0, 0]
        self.target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        
        self.grid[self.target_pos[0], self.target_pos[1]] = 2
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        
        self.message = "Вижу цель! Иду на сближение."
        return self._get_obs(), {}

    def _get_obs(self):
        return self.grid.copy()

    def step(self, action):
        # Очищаем старую позицию агента
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
        
        # Вычисляем новую позицию
        new_pos = list(self.agent_pos)
        if action == 0: new_pos[0] -= 1   # Вверх
        elif action == 1: new_pos[1] += 1 # Вправо
        elif action == 2: new_pos[0] += 1 # Вниз
        elif action == 3: new_pos[1] -= 1 # Влево

        reward = -0.1 # Небольшой штраф за каждый шаг (чтобы искал короткий путь)
        terminated = False
        
        # Проверка столкновений со стенами поля и препятствиями
        if (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and 
            self.grid[new_pos[0], new_pos[1]] != 1):
            
            self.agent_pos = new_pos
            self.message = "Путь чист, двигаюсь вперед."
        else:
            reward = -1.0 # Штраф за удар о стену
            self.message = "Ой! Путь заблокирован. Ищу обход..."

        # Проверка победы
        if self.agent_pos == self.target_pos:
            reward = 10.0
            terminated = True
            self.message = "Ура! Я добрался до цели!"

        # Ставим агента на новую позицию
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        """Отрисовка графики и обработка кликов мыши"""
        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Нейро-Агент в лабиринте")
            self.font = pygame.font.SysFont('Arial', 24)
            self.clock = pygame.time.Clock()

        # Обработка событий Pygame (клики мышкой для стен)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Получаем координаты клика
                x, y = pygame.mouse.get_pos()
                col = x // CELL_SIZE
                row = y // CELL_SIZE
                
                # Ставим/убираем стену (если это не цель и не агент)
                if row < GRID_SIZE and self.grid[row, col] not in [2, 3]:
                    self.grid[row, col] = 1 if self.grid[row, col] == 0 else 0
                    self.message = "Кто-то меняет лабиринт!"

        self.window.fill(WHITE)

        # Рисуем сетку и объекты
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.window, GRAY, rect, 1)
                
                cell_val = self.grid[row, col]
                if cell_val == 1:   # Стена
                    pygame.draw.rect(self.window, BLACK, rect)
                elif cell_val == 2: # Цель
                    pygame.draw.rect(self.window, GREEN, rect)
                elif cell_val == 3: # Агент
                    pygame.draw.circle(self.window, BLUE, rect.center, CELL_SIZE // 2 - 5)

        # Рисуем панель мыслей агента
        pygame.draw.rect(self.window, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        text_surface = self.font.render(self.message, True, BLACK)
        self.window.blit(text_surface, (10, SCREEN_HEIGHT - 70))

        pygame.display.flip()
        self.clock.tick(10) # Ограничиваем скорость обновления

# --- 2. ГЛАВНЫЙ БЛОК ---
if __name__ == "__main__":
    # Создаем среду
    env = MazeEnv()
    
    print("🚀 Начинаю обучение нейросети... Подожди минутку.")
    # Используем алгоритм PPO (многослойный персептрон 'MlpPolicy')
    model = PPO("MlpPolicy", env, verbose=0)
    
    # Обучаем модель (20 000 шагов для базовой умности)
    model.learn(total_timesteps=20000)
    print("✅ Обучение завершено! Запускаю симуляцию.")

    # Интерактивная демонстрация
    obs, info = env.reset()
    
    while True:
        env.render()
        
        # Нейросеть предсказывает лучшее действие
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.3) 
        
        if terminated:
            env.render()
            time.sleep(2) # Пауза после победы
            obs, info = env.reset()

#TODO add new part of code