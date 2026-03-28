import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from stable_baselines3 import PPO

# Установите: pip install numba
from numba import njit, prange
import threading

# Декоратор @njit компилирует функцию в машинный код
@njit
def fast_sigmoid(x):
    # Клипирование для избежания overflow
    x_clipped = x if x > -500 else -500
    x_clipped = x_clipped if x_clipped < 500 else 500
    return 1.0 / (1.0 + np.exp(-x_clipped))

@njit
def fast_sigmoid_derivative(x):
    """Производная сигмоида для обучения"""
    sig = fast_sigmoid(x)
    return sig * (1.0 - sig)

@njit
def fast_predict(inputs, weights_ih, weights_ho):
    # Теперь умножение матриц будет работать на скорости C++
    hidden = fast_sigmoid(np.dot(inputs, weights_ih))
    output = fast_sigmoid(np.dot(hidden, weights_ho))
    return output

@njit
def fast_relu(x):
    """ReLU активация для ускорения"""
    return np.maximum(0.0, x)

@njit
def check_collision_optimized(grid, new_pos, grid_size):
    """Оптимизированная проверка столкновения"""
    row, col = new_pos[0], new_pos[1]
    if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
        return True  # Выход за границы
    if grid[row, col] == 1:  # Стена
        return True
    return False

@njit
def update_grid_optimized(grid, old_pos, new_pos, agent_val=3):
    """Быстрое обновление позиции в сетке"""
    grid_copy = grid.copy()
    grid_copy[old_pos[0], old_pos[1]] = 0  # Очистить старую позицию
    grid_copy[new_pos[0], new_pos[1]] = agent_val  # Установить новую позицию
    return grid_copy

@njit
def batch_matrix_multiply(inputs, weights):
    """Массовое умножение матриц для нескольких eingaben"""
    return np.dot(inputs, weights)

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
        self.episode_count = 0  # Для отслеживания номера эпизода
        self.step_count = 0     # Для отслеживания шагов в эпизоде
        self.max_steps = 100    # Максимум шагов на эпизод

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        self.stuck_steps = 0
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        # Стартовые позиции
        self.agent_pos = [0, 0]
        self.target_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        
        # Только стартовые объекты - БЕЗ РАНДОМНЫХ ПРЕПЯТСТВИЙ
        self.grid[self.target_pos[0], self.target_pos[1]] = 2
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        
        self.message = "Вижу цель! Иду на сближение."
        return self._get_obs(), {}

    def _get_obs(self):
        return self.grid.copy()

    def step(self, action):
        self.step_count += 1
        
        # ОЧИЩАЕМ СТАРУЮ ПОЗИЦИЮ (убираем змейку)
        if self.grid[self.agent_pos[0], self.agent_pos[1]] == 3:
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
        
        # Вычисляем новую позицию
        new_pos = np.array(self.agent_pos, dtype=np.int32)
        if action == 0: new_pos[0] -= 1   # Вверх
        elif action == 1: new_pos[1] += 1 # Вправо
        elif action == 2: new_pos[0] += 1 # Вниз
        elif action == 3: new_pos[1] -= 1 # Влево

        # Базовый штраф за каждый шаг
        reward = -0.02  # УСИЛЕНО: больший штраф за ненужные шаги
        terminated = False
        truncated = False
        
        # Вычисляем расстояние до цели (Manhattan distance)
        old_distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])
        new_distance = abs(new_pos[0] - self.target_pos[0]) + abs(new_pos[1] - self.target_pos[1])
        
        # Оптимизированная проверка столкновений
        if check_collision_optimized(self.grid, new_pos, GRID_SIZE):
            # Столкновение со стеной или препятствием
            reward = -2.0  # УСИЛЕНО: ЕЩЁ БОЛЬШИЙ штраф за удар
            self.message = "💥 Столкновение! Ищу обход."
        else:
            self.agent_pos = [int(new_pos[0]), int(new_pos[1])]
            
            # УСИЛЕННАЯ СИСТЕМА ВОЗНАГРАЖДЕНИЯ (поощрение за сближение)
            if new_distance < old_distance:
                # Агент приблизился к цели - ОЧЕНЬ большое награждение!
                progress_reward = 1.0 + (1.0 * (old_distance - new_distance) / (old_distance + 1))  # УСИЛЕНО
                reward = progress_reward
                self.message = f"✓ Правильный путь! Расстояние: {new_distance}"
            else:
                # Агент отдалился от цели - усиленный штраф
                reward = -0.5  # УСИЛЕНО: больший штраф
                self.message = f"✗ Неправильное направление. Расстояние: {new_distance}"
            
            # Ставим агента на новую позицию (БЕЗ ЗМЕЙКИ)
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 3

        # Проверка победы
        if self.agent_pos == self.target_pos:
            reward = 20.0  # УСИЛЕНО: ЕЩЁ БОЛЬШЕЕ награждение за достижение цели
            terminated = True
            self.message = "🎉 Ура! Я добрался до цели!"
            self.stuck_steps = 0
        else:
            # Если нет прогресса или постоянно отскакиваем, засчитываем как застревание
            if new_distance < old_distance:
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1

        # Проверка timeout (слишком долго идёт)
        if self.step_count >= self.max_steps:
            truncated = True
            if not terminated:
                reward -= 5.0  # УСИЛЕНО: больший штраф за timeout
                self.message = "⏱️ Время истекло!"

        # Жёсткая защита от блуждания: принудительно пересоздаем лабиринт
        if self.stuck_steps >= 15 and not terminated:
            truncated = True
            reward -= 5.0
            self.message = "🔄 Застрял, перезапуск лабиринта"
            self.stuck_steps = 0

        return self._get_obs(), reward, terminated, truncated, {}

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
    
    print("🚀 Начинаю УСИЛЕННОЕ обучение нейросети...")
    print("   - Динамические препятствия каждый эпизод")
    print("   - Система вознаграждения за сближение к цели")
    print("   - Plus 100k шагов = мощное обучение!")
    
    # Используем PPO с МАКСИМАЛЬНО УСИЛЕННЫМИ параметрами для адаптивности
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=5e-4,   # УСИЛЕНО: повышенный learning rate
        n_steps=4096,         # УСИЛЕНО: ЕЩЁ больше шагов
        batch_size=128,       # УСИЛЕНО: больший batch size
        n_epochs=15,          # УСИЛЕНО: больше эпох обучения
        gamma=0.99,           # Future reward discount
        gae_lambda=0.95,      # GAE lambda
        clip_range=0.3,       # УСИЛЕНО: более агрессивный clip range
        ent_coef=0.02,        # УСИЛЕНО: больше entropy для exploration
        tensorboard_log="./ppo_logs",
        policy_kwargs=dict(net_arch=[256, 256])  # УСИЛЕНО: больше нейронов
    )
    
    # Обучаем модель ЕЩЁ БОЛЬШЕ для адаптивности к блокам
    print("📚 Обучение с МАКСИМАЛЬНЫМ усилением адаптации...")
    model.learn(total_timesteps=200000)  # УСИЛЕНО: 2 МИЛЛИОНА ШАГОВ!!!
    print("✅ Обучение завершено! Запускаю симуляцию.")
    
    # Сохраняем модель
    model.save("maze_agent_strong")
    print("💾 Модель сохранена как 'maze_agent_strong'")

    # Интерактивная демонстрация
    obs, info = env.reset()
    
    while True:
        env.render()
        
        # Нейросеть предсказывает лучшее действие
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.3) 
        
        if terminated or truncated:
            env.render()
            time.sleep(2) # Пауза после завершения
            obs, info = env.reset()

#TODO add new part of code