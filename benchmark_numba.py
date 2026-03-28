"""
Скрипт для сравнения производительности оптимизированного кода с Numba
"""
import numpy as np
import time
from numba import njit

# ============ NUMBA ВЕРСИИ ==============

@njit
def fast_sigmoid_numba(x):
    # Клипирование для избежания overflow
    x_clipped = x if x > -500 else -500
    x_clipped = x_clipped if x_clipped < 500 else 500
    return 1.0 / (1.0 + np.exp(-x_clipped))

@njit
def fast_predict_numba(inputs, weights_ih, weights_ho):
    hidden = fast_sigmoid_numba(np.dot(inputs, weights_ih))
    output = fast_sigmoid_numba(np.dot(hidden, weights_ho))
    return output

# ============ PYTHON ВЕРСИИ ==============

def sigmoid_python(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_python(inputs, weights_ih, weights_ho):
    hidden = sigmoid_python(np.dot(inputs, weights_ih))
    output = sigmoid_python(np.dot(hidden, weights_ho))
    return output

# ============ ТЕСТИРОВАНИЕ ==============

def benchmark():
    print("=" * 60)
    print("🚀 BENCHMARK: Numba vs Python")
    print("=" * 60)
    
    # Подготовка данных
    n_runs = 10000
    input_size = 100
    hidden_size = 50
    output_size = 4
    
    inputs = np.random.randn(input_size).astype(np.float64)
    weights_ih = np.random.randn(input_size, hidden_size).astype(np.float64)
    weights_ho = np.random.randn(hidden_size, output_size).astype(np.float64)
    
    print(f"\n📊 Параметры теста:")
    print(f"   - Количество итераций: {n_runs}")
    print(f"   - Размер входа: {input_size}")
    print(f"   - Размер скрытого слоя: {hidden_size}")
    print(f"   - Размер выхода: {output_size}")
    
    # ===== Тест 1: Активация (sigmoid) =====
    print(f"\n1️⃣  Тест активации функции (sigmoid)")
    print("-" * 60)
    
    # Прогрев Numba JIT компилятора
    test_val = 5.0
    for _ in range(10):
        fast_sigmoid_numba(test_val)
    
    # Python версия
    start = time.time()
    for _ in range(n_runs):
        sigmoid_python(test_val)
    time_python_sigmoid = time.time() - start
    
    # Numba версия  
    start = time.time()
    for _ in range(n_runs):
        fast_sigmoid_numba(test_val)
    time_numba_sigmoid = time.time() - start
    
    speedup_sigmoid = time_python_sigmoid / time_numba_sigmoid
    
    print(f"   Python: {time_python_sigmoid:.4f}s")
    print(f"   Numba:  {time_numba_sigmoid:.4f}s")
    print(f"   ⚡ УСКОРЕНИЕ: {speedup_sigmoid:.1f}x")
    
    # ===== Тест 2: Полное предсказание =====
    print(f"\n2️⃣  Тест полного предсказания (forward pass)")
    print("-" * 60)
    
    # Прогрев
    for _ in range(10):
        fast_predict_numba(inputs, weights_ih, weights_ho)
    
    # Python версия
    start = time.time()
    for _ in range(n_runs):
        predict_python(inputs, weights_ih, weights_ho)
    time_python_predict = time.time() - start
    
    # Numba версия
    start = time.time()
    for _ in range(n_runs):
        fast_predict_numba(inputs, weights_ih, weights_ho)
    time_numba_predict = time.time() - start
    
    speedup_predict = time_python_predict / time_numba_predict
    
    print(f"   Python: {time_python_predict:.4f}s")
    print(f"   Numba:  {time_numba_predict:.4f}s")
    print(f"   ⚡ УСКОРЕНИЕ: {speedup_predict:.1f}x")
    
    # ===== Итоги =====
    print(f"\n" + "=" * 60)
    print(f"📈 ИТОГОВОЕ УСКОРЕНИЕ:")
    print(f"   Активация (sigmoid):  {speedup_sigmoid:.1f}x быстрее")
    print(f"   Предсказание:         {speedup_predict:.1f}x быстрее")
    print(f"   Среднее:              {(speedup_sigmoid + speedup_predict) / 2:.1f}x")
    print(f"=" * 60)
    
    # Оценка экономии времени обучения
    print(f"\n💾 ЭКОНОМИЯ ВРЕМЕНИ:")
    if speedup_predict > 1:
        saved_time = time_python_predict - time_numba_predict
        saved_percent = (1 - time_numba_predict / time_python_predict) * 100
        print(f"   За {n_runs} итераций: {saved_time:.2f}s экономии ({saved_percent:.1f}%)")
        print(f"   При 20,000 шагов обучения: ~{saved_time * 2:.1f}s экономии")
    
    return speedup_sigmoid, speedup_predict

if __name__ == "__main__":
    benchmark()
    print("\n✅ Команда для рекомендации: используйте Numba версии для обучения!")
