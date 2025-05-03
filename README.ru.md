# Глава 46: Сети Временного Внимания для Финансовых Временных Рядов

В этой главе рассматриваются **Сети Временного Внимания** (Temporal Attention Networks) — специализированные механизмы внимания, предназначенные для захвата временных зависимостей в финансовых данных. Мы фокусируемся на архитектуре **TABL (Temporal Attention-Augmented Bilinear Layer)** и её вариантах, которые доказали высокую эффективность для предсказания движений рынка на основе данных книги заявок (LOB) и других временных рядов.

<p align="center">
<img src="https://i.imgur.com/Zy8R4qF.png" alt="Диаграмма архитектуры TABL" width="70%">
</p>

## Содержание

1. [Введение во Временное Внимание](#введение-во-временное-внимание)
    * [Зачем Временное Внимание?](#зачем-временное-внимание)
    * [Ключевые Преимущества](#ключевые-преимущества)
    * [Сравнение с Другими Моделями](#сравнение-с-другими-моделями)
2. [Архитектура TABL](#архитектура-tabl)
    * [Билинейная Проекция](#билинейная-проекция)
    * [Механизм Временного Внимания](#механизм-временного-внимания)
    * [BL (Билинейный Слой)](#bl-билинейный-слой)
    * [TABL (Временной Внимательный Билинейный Слой)](#tabl-временной-внимательный-билинейный-слой)
3. [Многоголовое Временное Внимание](#многоголовое-временное-внимание)
    * [Многоголовый TABL](#многоголовый-tabl)
    * [Параллельные Головы Внимания](#параллельные-головы-внимания)
4. [Обработка Данных](#обработка-данных)
    * [Признаки Книги Заявок](#признаки-книги-заявок)
    * [Признаки OHLCV](#признаки-ohlcv)
    * [Инженерия Признаков](#инженерия-признаков)
5. [Практические Примеры](#практические-примеры)
    * [01: Подготовка Данных](#01-подготовка-данных)
    * [02: Архитектура TABL](#02-архитектура-tabl)
    * [03: Обучение Модели](#03-обучение-модели)
    * [04: Визуализация Внимания](#04-визуализация-внимания)
    * [05: Торговая Стратегия](#05-торговая-стратегия)
6. [Реализация на Rust](#реализация-на-rust)
7. [Реализация на Python](#реализация-на-python)
8. [Лучшие Практики](#лучшие-практики)
9. [Ресурсы](#ресурсы)

## Введение во Временное Внимание

Сети Временного Внимания созданы для решения фундаментальной задачи в финансовом прогнозировании: **какие прошлые события наиболее важны для предсказания будущего?**

В отличие от стандартных рекуррентных моделей, которые обрабатывают все временные шаги одинаково, механизмы временного внимания учатся **фокусироваться на наиболее информативных моментах** во входной последовательности.

### Зачем Временное Внимание?

Традиционные модели обрабатывают все временные шаги одинаково:

```text
Время:    t-5  t-4  t-3  t-2  t-1  t
Вес:       1    1    1    1    1   1
          (Все события одинаково важны)
```

Временное Внимание обучает адаптивные веса:

```text
Время:    t-5  t-4  t-3  t-2  t-1  t
Вес:      0.05 0.10 0.40 0.30 0.10 0.05
          (Внимание фокусируется на t-3 и t-2)
```

**Ключевая идея**: На финансовых рынках определённые моменты имеют непропорционально большое значение — крупные сделки, резкие всплески волатильности или специфические паттерны часто предшествуют движениям цены. Временное внимание автоматически учится выявлять эти критические моменты.

### Ключевые Преимущества

1. **Автоматический Отбор Признаков во Времени**
   - Обучается определять релевантные временные шаги
   - Не требует ручной разработки признаков "важных моментов"
   - Адаптируется к различным рыночным условиям

2. **Интерпретируемость**
   - Веса внимания показывают, какие события повлияли на предсказания
   - Полезно для понимания решений модели
   - Позволяет проводить апостериорный анализ торговых сигналов

3. **Вычислительная Эффективность**
   - TABL имеет сложность O(T·D) против O(T²·D) для self-attention
   - Намного быстрее LSTM для длинных последовательностей
   - Меньшие требования к памяти

4. **Высокая Производительность**
   - Превосходит LSTM и CNN в задачах предсказания LOB
   - Достигает state-of-the-art на бенчмарке FI-2010
   - Эффективна всего с 1-2 слоями

### Сравнение с Другими Моделями

| Характеристика | LSTM | CNN | Transformer | TABL |
|----------------|------|-----|-------------|------|
| Временное внимание | ✗ | ✗ | ✓ (self) | ✓ (обучаемое) |
| Сложность | O(T·D²) | O(T·K·D) | O(T²·D) | O(T·D) |
| Интерпретируемость | Низкая | Низкая | Средняя | Высокая |
| Предсказание LOB | Хорошо | Хорошо | Хорошо | Лучше всех |
| Эффективность памяти | ✗ | ✓ | ✗ | ✓ |
| Мало слоёв нужно | ✗ | ✗ | ✗ | ✓ |

## Архитектура TABL

Архитектура TABL объединяет **билинейные проекции** с **временным вниманием** для создания эффективной и интерпретируемой модели.

```text
┌──────────────────────────────────────────────────────────────────────┐
│                    СЕТЬ ВРЕМЕННОГО ВНИМАНИЯ                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Вход: X ∈ ℝ^(T×D)                                                   │
│  (T временных шагов, D признаков)                                    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Билинейная Проекция (BL)                        │        │
│  │                                                           │        │
│  │   H = σ(W₁ · X · W₂ + b)                                 │        │
│  │                                                           │        │
│  │   W₁ ∈ ℝ^(T'×T)  - Временная проекция                    │        │
│  │   W₂ ∈ ℝ^(D×D')  - Проекция признаков                    │        │
│  │   H ∈ ℝ^(T'×D')  - Сжатое представление                  │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Временное Внимание (TA)                         │        │
│  │                                                           │        │
│  │   α = softmax(w · tanh(U · X^T))                         │        │
│  │   c = X^T · α                                             │        │
│  │                                                           │        │
│  │   α ∈ ℝ^T         - Веса внимания                        │        │
│  │   c ∈ ℝ^D         - Контекстный вектор                   │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Выходной Слой                                   │        │
│  │                                                           │        │
│  │   y = softmax(W_out · flatten(H, c) + b_out)             │        │
│  │                                                           │        │
│  │   3 класса: Вверх / Без изменений / Вниз                 │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Билинейная Проекция

Билинейный слой выполняет два одновременных линейных преобразования:

```python
class BilinearLayer(nn.Module):
    """
    Билинейная проекция: H = σ(W₁ · X · W₂ + b)

    Преобразует (T, D) → (T', D') путём проекции
    как временного, так и признакового измерений одновременно.
    """

    def __init__(self, T_in, T_out, D_in, D_out, dropout=0.1):
        super().__init__()
        # Временная проекция: (T_out, T_in)
        self.W1 = nn.Parameter(torch.randn(T_out, T_in) * 0.01)
        # Проекция признаков: (D_in, D_out)
        self.W2 = nn.Parameter(torch.randn(D_in, D_out) * 0.01)
        # Смещение: (T_out, D_out)
        self.bias = nn.Parameter(torch.zeros(T_out, D_out))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, T_in, D_in)
        # W1·X: (batch, T_out, D_in)
        out = torch.matmul(self.W1, x)
        # W1·X·W2: (batch, T_out, D_out)
        out = torch.matmul(out, self.W2)
        out = out + self.bias
        out = self.activation(out)
        return self.dropout(out)
```

**Почему Билинейный?**
- Захватывает взаимодействия между временем и признаками
- Более выразителен, чем простые линейные слои
- Эффективно снижает размерность по обеим осям

### Механизм Временного Внимания

Временное внимание вычисляет взвешенную сумму по временным шагам:

```python
class TemporalAttention(nn.Module):
    """
    Временное внимание: α = softmax(w · tanh(U · X^T))

    Обучается фокусироваться на важных временных шагах.
    """

    def __init__(self, D, attention_dim=64):
        super().__init__()
        # Проекция признаков в пространство внимания
        self.U = nn.Linear(D, attention_dim, bias=False)
        # Вектор запроса внимания
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, T, D)
        # Вычисление оценок внимания
        h = torch.tanh(self.U(x))          # (batch, T, attention_dim)
        scores = self.w(h).squeeze(-1)      # (batch, T)

        # Softmax по временному измерению
        alpha = F.softmax(scores, dim=-1)   # (batch, T)

        # Взвешенная сумма: контекстный вектор
        context = torch.bmm(
            alpha.unsqueeze(1),             # (batch, 1, T)
            x                                # (batch, T, D)
        ).squeeze(1)                         # (batch, D)

        return context, alpha
```

**Интерпретация:**
- `alpha[t]` указывает на важность временного шага t
- Высокое внимание на конкретных событиях показывает фокус модели
- Контекстный вектор `c` обобщает последовательность

### BL (Билинейный Слой)

BL слой — это версия без внимания:

```python
class BL(nn.Module):
    """Билинейный слой без внимания"""

    def __init__(self, config):
        super().__init__()
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )

    def forward(self, x):
        h = self.bilinear(x)
        return h.flatten(1)  # (batch, hidden_T * hidden_D)
```

### TABL (Временной Внимательный Билинейный Слой)

Полный TABL объединяет оба компонента:

```python
class TABL(nn.Module):
    """
    Temporal Attention-Augmented Bilinear Layer

    Объединяет билинейную проекцию с временным вниманием.
    """

    def __init__(self, config):
        super().__init__()
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )
        self.attention = TemporalAttention(
            config.input_dim,
            config.attention_dim
        )

    def forward(self, x, return_attention=False):
        # Билинейная проекция
        h = self.bilinear(x)  # (batch, hidden_T, hidden_D)
        h_flat = h.flatten(1)  # (batch, hidden_T * hidden_D)

        # Временное внимание
        context, alpha = self.attention(x)  # (batch, D), (batch, T)

        # Конкатенация
        out = torch.cat([h_flat, context], dim=-1)

        if return_attention:
            return out, alpha
        return out
```

## Многоголовое Временное Внимание

Расширение TABL несколькими головами внимания позволяет модели фокусироваться на разных аспектах одновременно:

### Многоголовый TABL

```python
class MultiHeadTABL(nn.Module):
    """
    Multi-Head Temporal Attention Bilinear Layer

    Использует несколько голов внимания для захвата
    различных временных паттернов в данных.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads

        # Общая билинейная проекция
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )

        # Несколько голов внимания
        self.attention_heads = nn.ModuleList([
            TemporalAttention(config.input_dim, config.attention_dim)
            for _ in range(config.n_heads)
        ])

        # Объединение голов
        self.head_combine = nn.Linear(
            config.n_heads * config.input_dim,
            config.input_dim
        )

    def forward(self, x, return_attention=False):
        # Билинейная проекция
        h = self.bilinear(x)
        h_flat = h.flatten(1)

        # Многоголовое внимание
        contexts = []
        alphas = []
        for head in self.attention_heads:
            ctx, alpha = head(x)
            contexts.append(ctx)
            alphas.append(alpha)

        # Объединение голов
        multi_context = torch.cat(contexts, dim=-1)
        combined = self.head_combine(multi_context)

        # Финальный выход
        out = torch.cat([h_flat, combined], dim=-1)

        if return_attention:
            return out, torch.stack(alphas, dim=1)  # (batch, n_heads, T)
        return out
```

### Параллельные Головы Внимания

Каждая голова может фокусироваться на разных паттернах:
- **Голова 1**: Краткосрочные движения цены
- **Голова 2**: Всплески объёма
- **Голова 3**: Дисбалансы книги заявок
- **Голова 4**: Трендовые паттерны

```text
Визуализация Многоголового Внимания:

Время:    t-10  t-9   t-8   t-7   t-6   t-5   t-4   t-3   t-2   t-1
Голова 1: ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ██▓▓  ████  ██▓▓  ░░░░  ░░░░
          (Фокус на среднесрочных событиях)

Голова 2: ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ████  ████
          (Фокус на недавних событиях)

Голова 3: ████  ░░░░  ░░░░  ████  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░
          (Фокус на периодических паттернах)

Объединённо: ▓▓▓▓  ░░░░  ░░░░  ▓▓▓▓  ░░░░  ▓▓▓▓  ████  ▓▓▓▓  ████  ████
            (Агрегированное внимание)
```

## Обработка Данных

### Признаки Книги Заявок

Данные LOB предоставляют богатую информацию о микроструктуре рынка:

```python
def extract_lob_features(lob_snapshot):
    """
    Извлечение признаков из снимка книги заявок.

    Возвращает признаки для входа TABL.
    """
    features = {}

    # Ценовые уровни (обычно 10 уровней с каждой стороны)
    features['ask_prices'] = lob_snapshot['asks'][:, 0]  # (10,)
    features['bid_prices'] = lob_snapshot['bids'][:, 0]  # (10,)
    features['ask_volumes'] = lob_snapshot['asks'][:, 1]  # (10,)
    features['bid_volumes'] = lob_snapshot['bids'][:, 1]  # (10,)

    # Производные признаки
    mid_price = (features['ask_prices'][0] + features['bid_prices'][0]) / 2
    spread = features['ask_prices'][0] - features['bid_prices'][0]

    # Дисбаланс ордеров
    total_ask_vol = features['ask_volumes'].sum()
    total_bid_vol = features['bid_volumes'].sum()
    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    features['mid_price'] = mid_price
    features['spread'] = spread
    features['imbalance'] = imbalance

    return features
```

### Признаки OHLCV

Для криптовалютных/биржевых данных с использованием OHLCV:

```python
def extract_ohlcv_features(df):
    """
    Извлечение признаков из данных OHLCV.
    """
    features = pd.DataFrame()

    # Ценовые признаки
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_open_diff'] = (df['close'] - df['open']) / df['open']

    # Объёмные признаки
    features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

    # Технические индикаторы
    features['rsi'] = compute_rsi(df['close'], 14)
    features['macd'] = compute_macd(df['close'])

    # Волатильность
    features['volatility'] = features['log_return'].rolling(20).std()

    return features.dropna()
```

### Инженерия Признаков

Рекомендуемые признаки для моделей с временным вниманием:

| Признак | Описание | Важность |
|---------|----------|----------|
| `log_return` | Логарифмическое изменение цены | Высокая |
| `spread` | Спред bid-ask | Высокая |
| `imbalance` | Дисбаланс книги заявок | Высокая |
| `volume_ratio` | Объём к скользящему среднему | Средняя |
| `volatility` | Скользящая волатильность | Средняя |
| `price_levels` | Ценовые уровни LOB | Средняя |
| `volume_levels` | Объёмные уровни LOB | Средняя |

## Практические Примеры

### 01: Подготовка Данных

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import Tuple, List

def prepare_tabl_data(
    df: pd.DataFrame,
    lookback: int = 100,
    horizon: int = 10,
    threshold: float = 0.0002
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготовка данных для обучения модели TABL.

    Args:
        df: DataFrame с данными OHLCV
        lookback: Количество временных шагов назад
        horizon: Горизонт прогнозирования
        threshold: Порог для классификации направления

    Returns:
        X: Признаки (n_samples, lookback, n_features)
        y: Метки (n_samples,) - 0: вниз, 1: без изменений, 2: вверх
    """
    # Извлечение признаков
    features = extract_features(df)

    # Нормализация признаков
    features_norm = (features - features.mean()) / features.std()

    # Создание последовательностей
    X, y = [], []

    for i in range(lookback, len(features_norm) - horizon):
        # Входная последовательность
        x_seq = features_norm.iloc[i-lookback:i].values

        # Цель: будущая доходность
        future_return = np.log(
            df['close'].iloc[i + horizon] / df['close'].iloc[i]
        )

        # Классификация направления
        if future_return > threshold:
            label = 2  # Вверх
        elif future_return < -threshold:
            label = 0  # Вниз
        else:
            label = 1  # Без изменений

        X.append(x_seq)
        y.append(label)

    return np.array(X), np.array(y)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение признаков из данных OHLCV."""
    features = pd.DataFrame(index=df.index)

    # Доходности
    features['return'] = np.log(df['close'] / df['close'].shift(1))

    # Позиция цены
    features['hl_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # Объём
    vol_ma = df['volume'].rolling(20).mean()
    features['vol_ratio'] = df['volume'] / vol_ma

    # Волатильность
    features['volatility'] = features['return'].rolling(20).std()

    # Моментум
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    return features.dropna()
```

### 02: Архитектура TABL

См. [python/model.py](python/model.py) для полной реализации.

### 03: Обучение Модели

```python
# python/03_train_model.py

import torch
from model import TABLModel, TABLConfig

# Конфигурация
config = TABLConfig(
    seq_len=100,
    input_dim=6,
    hidden_T=20,
    hidden_D=32,
    attention_dim=64,
    n_heads=4,
    n_classes=3,
    dropout=0.2
)

# Инициализация модели
model = TABLModel(config)
print(f"Параметры: {sum(p.numel() for p in model.parameters()):,}")

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Цикл обучения
best_acc = 0
for epoch in range(100):
    model.train()
    train_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Прямой проход
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Обратный проход
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    # Валидация
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')

    print(f"Эпоха {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
```

### 04: Визуализация Внимания

```python
# python/04_attention_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, x, timestamps=None):
    """
    Визуализация весов временного внимания.

    Args:
        model: Обученная модель TABL
        x: Входная последовательность (1, T, D)
        timestamps: Опциональные временные метки для оси X
    """
    model.eval()
    with torch.no_grad():
        logits, attention = model(x, return_attention=True)

    # attention: (1, n_heads, T) или (1, T)
    attention = attention.squeeze(0).numpy()

    if attention.ndim == 1:
        # Одна голова
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(attention)), attention)
        ax.set_xlabel('Временной шаг')
        ax.set_ylabel('Вес внимания')
        ax.set_title('Веса временного внимания')
    else:
        # Много голов
        n_heads = attention.shape[0]
        fig, axes = plt.subplots(n_heads + 1, 1, figsize=(12, 3 * (n_heads + 1)))

        for i, ax in enumerate(axes[:-1]):
            ax.bar(range(attention.shape[1]), attention[i])
            ax.set_ylabel(f'Голова {i+1}')
            ax.set_title(f'Голова внимания {i+1}')

        # Объединённое внимание
        combined = attention.mean(axis=0)
        axes[-1].bar(range(len(combined)), combined)
        axes[-1].set_xlabel('Временной шаг')
        axes[-1].set_ylabel('Объединённое')
        axes[-1].set_title('Объединённое внимание (среднее)')

    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    plt.show()


def attention_heatmap(model, dataset, n_samples=50):
    """
    Создание тепловой карты паттернов внимания по выборкам.
    """
    model.eval()
    all_attention = []

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0)
            _, attention = model(x, return_attention=True)
            all_attention.append(attention.squeeze().numpy())

    attention_matrix = np.stack(all_attention)  # (n_samples, T)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        attention_matrix,
        cmap='Blues',
        xticklabels=5,
        yticklabels=5
    )
    plt.xlabel('Временной шаг')
    plt.ylabel('Выборка')
    plt.title('Паттерны внимания по выборкам')
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.show()
```

### 05: Торговая Стратегия

```python
# python/05_strategy.py

def backtest_tabl_strategy(
    model,
    test_data,
    df_prices,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.6
):
    """
    Бэктестинг торговой стратегии TABL.

    Args:
        model: Обученная модель TABL
        test_data: Тестовый датасет
        df_prices: Ценовые данные, согласованные с test_data
        initial_capital: Начальный капитал
        transaction_cost: Стоимость транзакции
        confidence_threshold: Мин. вероятность для сделки
    """
    model.eval()
    capital = initial_capital
    position = 0  # -1: шорт, 0: нет позиции, 1: лонг

    results = []

    with torch.no_grad():
        for i, (x, _) in enumerate(test_data):
            x = x.unsqueeze(0)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze()

            # Получение предсказания и уверенности
            pred = probs.argmax().item()
            confidence = probs.max().item()

            # Торговая логика
            if confidence >= confidence_threshold:
                if pred == 2 and position <= 0:  # Сигнал вверх
                    # Закрыть шорт, открыть лонг
                    if position == -1:
                        capital *= (1 - transaction_cost)
                    position = 1
                    capital *= (1 - transaction_cost)

                elif pred == 0 and position >= 0:  # Сигнал вниз
                    # Закрыть лонг, открыть шорт
                    if position == 1:
                        capital *= (1 - transaction_cost)
                    position = -1
                    capital *= (1 - transaction_cost)

            # Расчёт P&L
            if i > 0:
                price_return = df_prices['close'].iloc[i] / df_prices['close'].iloc[i-1] - 1
                capital *= (1 + position * price_return)

            results.append({
                'capital': capital,
                'position': position,
                'prediction': pred,
                'confidence': confidence
            })

    return pd.DataFrame(results)


def calculate_metrics(results_df, df_prices):
    """Расчёт метрик стратегии."""
    returns = results_df['capital'].pct_change().dropna()

    metrics = {
        'total_return': (results_df['capital'].iloc[-1] / results_df['capital'].iloc[0] - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24),  # Часовые данные
        'max_drawdown': ((results_df['capital'].cummax() - results_df['capital']) /
                         results_df['capital'].cummax()).max() * 100,
        'win_rate': (returns > 0).mean() * 100,
        'n_trades': (results_df['position'].diff() != 0).sum()
    }

    return metrics
```

## Реализация на Rust

См. [rust/](rust/) для полной реализации на Rust с использованием данных Bybit.

```text
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Основные экспорты библиотеки
│   ├── api/                # Клиент Bybit API
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP клиент для Bybit
│   │   └── types.rs        # Типы ответов API
│   ├── data/               # Обработка данных
│   │   ├── mod.rs
│   │   ├── loader.rs       # Утилиты загрузки данных
│   │   ├── features.rs     # Инженерия признаков
│   │   └── dataset.rs      # Датасет для обучения
│   ├── model/              # Архитектура TABL
│   │   ├── mod.rs
│   │   ├── bilinear.rs     # Билинейный слой
│   │   ├── attention.rs    # Временное внимание
│   │   └── tabl.rs         # Полная модель TABL
│   └── strategy/           # Торговая стратегия
│       ├── mod.rs
│       ├── signals.rs      # Генерация сигналов
│       └── backtest.rs     # Движок бэктестинга
└── examples/
    ├── fetch_data.rs       # Загрузка данных Bybit
    ├── train.rs            # Обучение модели
    └── backtest.rs         # Запуск бэктеста
```

### Быстрый Старт (Rust)

```bash
# Перейти в директорию Rust проекта
cd rust

# Загрузить данные с Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Обучить модель
cargo run --example train -- --epochs 100 --batch-size 32

# Запустить бэктест
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

См. [python/](python/) для реализации на Python.

```text
python/
├── model.py                # Реализация модели TABL
├── data.py                 # Загрузка данных и признаки
├── train.py                # Скрипт обучения
├── strategy.py             # Торговая стратегия
├── example_usage.py        # Пример использования
├── requirements.txt        # Зависимости
└── __init__.py
```

### Быстрый Старт (Python)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели
python train.py --symbols BTCUSDT --epochs 100

# Запуск бэктеста
python strategy.py --model checkpoints/best_model.pt
```

## Лучшие Практики

### Когда Использовать TABL

**Хорошие сценарии использования:**
- Предсказание движения средней цены LOB
- Высокочастотное прогнозирование направления
- Краткосрочные прогнозы (секунды-минуты)
- Требования к интерпретируемости

**Не идеально для:**
- Очень длинных последовательностей (>500 временных шагов)
- Многогоризонтного прогнозирования
- Распределения портфеля (используйте Stockformer)

### Рекомендации по Гиперпараметрам

| Параметр | Рекомендуется | Примечания |
|----------|---------------|------------|
| `seq_len` | 50-200 | Зависит от частоты данных |
| `hidden_T` | 10-30 | Временное сжатие |
| `hidden_D` | 32-128 | Сжатие признаков |
| `attention_dim` | 32-128 | Соответствовать hidden_D |
| `n_heads` | 2-8 | Больше для сложных паттернов |
| `dropout` | 0.1-0.3 | Выше для малых данных |

### Распространённые Ошибки

1. **Дисбаланс классов**: Используйте взвешенную функцию потерь или пересэмплирование
2. **Переобучение**: Применяйте dropout, раннюю остановку
3. **Масштабирование признаков**: Нормализуйте входы к нулевому среднему и единичной дисперсии
4. **Выбор порога**: Тщательно настраивайте пороги классификации

## Ресурсы

### Научные Работы

- [Temporal Attention augmented Bilinear Network for Financial Time-Series Data Analysis](https://arxiv.org/abs/1712.00975) — Оригинальная статья TABL
- [Multi-head Temporal Attention-Augmented Bilinear Network](https://ieeexplore.ieee.org/document/9909957/) — Многоголовое расширение
- [Augmented Bilinear Network for Incremental Multi-Stock Time-Series Classification](https://www.sciencedirect.com/science/article/pii/S0031320323003059) — Инкрементальное обучение

### Реализации

- [TABL Original (Python 2.7)](https://github.com/viebboy/TABL)
- [TABL PyTorch](https://github.com/LeonardoBerti00/TABL-Temporal-Attention-Augmented-Bilinear-Network-for-Financial-Time-Series-Data-Analysis)

### Связанные Главы

- [Глава 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Многогоризонтное прогнозирование
- [Глава 43: Stockformer Multivariate](../43_stockformer_multivariate) — Кросс-активное внимание
- [Глава 42: Dual Attention LOB](../42_dual_attention_lob) — Предсказание LOB

---

## Уровень Сложности

### Средний до продвинутого

Предварительные требования:
- Основы нейронных сетей
- Базовые понятия механизмов внимания
- Концепции прогнозирования временных рядов
- Библиотеки ML для PyTorch/Rust
