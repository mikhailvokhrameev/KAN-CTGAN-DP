##### <p align="left">RU version:</p>
---

# KAN-CTGAN-DP: Генерация синтетических табличных данных с использованием KAN-блоков и дифференциальной приватности

Данный репозиторий содержит исследовательский проект по разработке модифицированной генеративной модели на основе архитектуры **CTGAN** (Conditional Tabular Generative Adversarial Network). В рамках работы стандартные линейные остаточные блоки генератора были заменены на **слои сетей Колмогорова-Арнольда (KAN)** с B-сплайновой параметризацией, а в дискриминатор интегрированы механизмы **дифференциальной приватности** (DP-SGD).

Модель предназначена для создания синтетических банковских данных, которые статистически неотличимы от реальных, но при этом гарантируют защиту персональной информации.

<div align="center">

<img src="https://github.com/user-attachments/assets/50b0a00b-8bb7-4d0d-afc2-ca0171325e26" width="300"><br>
  
<em>Пример сети Колмогорова-Арнольда</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/25e45c6e-4369-4d3b-917b-506ae4f1ad3e" width="700"><br>
  
<em>Архитектурная схема исходного CTGAN-DP</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/d8ee6416-8a84-43c0-a8fb-93a15ecf690f" width="700"><br>
  
<em>Архитектурная схема представленного KAN-CTGAN-DP</em>
</div>

---

### Почему я делал этот проект?

Основная цель работы — преодоление критических барьеров в финансовой сфере:
*   **Правовые ограничения:** Требования Федерального закона № 152-ФЗ «О персональных данных» ограничивают обмен реальными данными о заёмщиках.
*   **Дисбаланс классов:** В кредитных выборках доля дефолтных заёмщиков обычно мала (2–10%), что требует инструментов для корректной аугментации данных с сохранением сложных нелинейных зависимостей.
*   **Технологический интерес:** Исследование того, как **KAN-блоки** могут улучшить аппроксимационную способность генератора по сравнению с классическими MLP-слоями при сохранении формальных гарантий приватности.

---

### Используемые технологии:

*   **Python 3.10+**
*   **PyTorch** — основной фреймворк для построения нейронных сетей.
*   **Opacus** — библиотека для реализации дифференциальной приватности через DP-SGD.
*   **KAN (Kolmogorov-Arnold Networks)** — инновационная архитектура со слоями `KANLinear`.
*   **CTGAN** — базовая модель для работы с табличными данными со смешанными типами признаков.
*   **NumPy / Pandas** — для обработки данных и статистического анализа.

---

### Структура проекта

*   `dataset/` — папка с данными для обучения и анализа (кредитный скоринг).
*   `main/` — основные скрипты для запуска обучения и оценки моделей.
*   `weights/` — сохраненные веса обученных моделей.
*   `presentation/` — файл презентации проекта.

---

### Установка

Для настройки среды проекта выполните следующее:

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/mikhailvokhrameev/KAN-CTGAN-DP.git
    ```
2.  **Создайте и активируйте виртуальное окружение.**
3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Использование

#### Шаг 1: Подготовка данных
Проект использует набор данных кредитного скоринга (например, Beeline Challenge). Данные проходят стадию **VGM-нормализации** (Variational Gaussian Mixture) для обработки непрерывных распределений.

#### Шаг 2: Тренировка модели (KAN-CTGAN-DP)
Запуск обучения гибридной архитектуры. В процессе используется оптимизатор **AdamW** и функция потерь **Binary Cross-Entropy**.
*   Генератор использует блоки `KAN_Residual` с активацией **SiLU** и нормализацией **LayerNorm**.
*   Дискриминатор обучается с использованием **DP-SGD**, где градиенты обрезаются и зашумляются для обеспечения бюджета приватности (например, $\epsilon = 5$).

#### Шаг 3: Оценка эффективности
Для оценки синтезированных данных используются три группы метрик:
*   **Utility:** Проверка эффективности на задаче классификации (ROC-AUC). Тестирование показало рост Utility на **2,43%** при использовании KAN.
*   **Fidelity:** Оценка точности воспроизведения распределений и корреляций (KL-Divergence, KS-Test).
*   **Privacy:** Проверка устойчивости к атакам на членство в выборке (MIA). Модель сохраняет максимальный уровень защиты (1.0000).

---

### Используемые внешние библиотеки и репозитории

*   **pykan** — реализация сетей Колмогорова-Арнольда.
*   **CTGAN** — базовая реализация условных GAN для таблиц от SDV.
*   **Opacus** — инструменты дифференциальной приватности от PyTorch.

---
##### <p align="left">ENG version:</p>
---

# KAN-CTGAN-DP: Synthetic Tabular Data Generation using KAN Blocks and Differential Privacy

This repository contains a research project on developing a modified generative model based on the **CTGAN** (Conditional Tabular Generative Adversarial Network) architecture. In this work, the standard linear residual blocks of the generator were replaced with **Kolmogorov-Arnold Network (KAN) layers** using B-spline parametrization, and **Differential Privacy** (DP-SGD) mechanisms were integrated into the discriminator.

The model is designed to create synthetic banking data that is statistically identical to real data while guaranteeing the protection of personal information.

<div align="center">

<img src="https://github.com/user-attachments/assets/50b0a00b-8bb7-4d0d-afc2-ca0171325e26" width="300"><br>
  
<em>Example of the Kolmogorov-Arnold Network</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/25e45c6e-4369-4d3b-917b-506ae4f1ad3e" width="700"><br>
  
<em>Architectural scheme of the original CTGAN-DP</em>
</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/d8ee6416-8a84-43c0-a8fb-93a15ecf690f" width="700"><br>
  
<em>Architectural scheme of the proposed KAN-CTGAN-DP</em>
</div>


---

### Why I did this project?

The main goal was to overcome critical barriers in the financial sector:
*   **Legal Restrictions:** Privacy laws (like Federal Law No. 152-FZ) limit the sharing of real borrower data.
*   **Class Imbalance:** Credit datasets typically have a low share of default borrowers (2–10%), requiring tools for correct data augmentation that preserve complex non-linear dependencies.
*   **Technological Interest:** Investigating how **KAN blocks** can improve the generator's approximation capability compared to classical MLP layers while maintaining formal privacy guarantees.

---

### Technologies used:

*   **Python 3.10+**
*   **PyTorch** — core framework for building neural networks.
*   **Opacus** — library for implementing Differential Privacy via DP-SGD.
*   **KAN (Kolmogorov-Arnold Networks)** — innovative architecture featuring `KANLinear` layers.
*   **CTGAN** — base model for handling tabular data with mixed feature types.
*   **NumPy / Pandas** — for data processing and statistical analysis.

---

### Project Structure

*   `dataset/` — contains data for training and analysis (credit scoring).
*   `main/` — core scripts for running training and evaluating models.
*   `weights/` — saved weights of trained models.
*   `presentation/` — project presentation file.

---

### Setup and Installation

Follow these steps to set up the project environment:
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mikhailvokhrameev/KAN-CTGAN-DP.git
    ```
2.  **Create and Activate a Virtual Environment.**
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Usage Workflow

#### Step 1: Prepare the Data
The project utilizes credit scoring datasets (e.g., Beeline Challenge). Data undergoes **VGM normalization** (Variational Gaussian Mixture) to handle continuous distributions.

#### Step 2: Train the Model (KAN-CTGAN-DP)
Train the hybrid architecture. The process uses the **AdamW** optimizer and **Binary Cross-Entropy** loss.
*   The Generator employs `KAN_Residual` blocks with **SiLU** activation and **LayerNorm**.
*   The Discriminator is trained using **DP-SGD**, where gradients are clipped and noised to satisfy a privacy budget (e.g., $\epsilon = 5$).

#### Step 3: Evaluate Performance
Three groups of metrics are used to assess the synthetic data:
*   **Utility:** Measures effectiveness on a classification task (ROC-AUC). Testing showed a **2.43%** increase in Utility when using KAN.
*   **Fidelity:** Evaluates statistical matching of distributions and correlations (KL-Divergence, KS-Test).
*   **Privacy:** Checks resistance to Membership Inference Attacks (MIA). The model maintains a maximum protection level (1.0000).

---

#### External libraries and repositories used

*   **pykan** — implementation of Kolmogorov-Arnold Networks.
*   **CTGAN** — base implementation of conditional GANs for tables by SDV.
*   **Opacus** — Differential Privacy tools by PyTorch.
