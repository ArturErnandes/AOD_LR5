# -*- coding: utf-8 -*-
import json
import math
from colorama import Fore, init
import matplotlib.pyplot as plt

DATA_FILE = "data.json"


# --------------------------- Чтение данных ---------------------------

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data["X"], data["Y"]


# --------------------------- Статистика ------------------------------

def compute_stats(x_list, y_list):
    n = len(x_list)

    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n

    var_x = sum((x - mean_x) ** 2 for x in x_list) / n
    var_y = sum((y - mean_y) ** 2 for y in y_list) / n

    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    cov_xy = sum((x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(n)) / n

    r_xy = cov_xy / (std_x * std_y)

    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "cov_xy": cov_xy,
        "r_xy": r_xy
    }


def interpret_correlation(r):
    r_abs = abs(r)

    if r_abs < 0.3:
        strength = "слабая"
    elif r_abs < 0.5:
        strength = "умеренная"
    elif r_abs < 0.7:
        strength = "заметная"
    elif r_abs < 0.9:
        strength = "сильная"
    else:
        strength = "очень сильная"

    direction = "прямая" if r > 0 else "обратная"

    return strength, direction


# --------------------------- Линейная регрессия ------------------------------

def compute_regression_params(stats):
    var_x = stats["std_x"] ** 2
    b = stats["cov_xy"] / var_x
    a = stats["mean_y"] - b * stats["mean_x"]
    return a, b


def compute_r2(r):
    return r ** 2


# --------------------------- График ------------------------------

def plot_regression(x_list, y_list, a, b):
    plt.figure("Корреляционное поле")

    plt.scatter(x_list, y_list, label="Данные", color="blue")

    x_min = min(x_list)
    x_max = max(x_list)
    x_line = [x_min, x_max]
    y_line = [a + b * x_min, a + b * x_max]

    plt.plot(x_line, y_line, label=f"Y = {a:.2f} + {b:.2f}X", color="red")

    plt.title("Линия регрессии")
    plt.xlabel("X — содержание компонента (%)")
    plt.ylabel("Y — твёрдость по Роквеллу")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------- MAIN ------------------------------

def main():
    x, y = load_data()

    print(f"{Fore.LIGHTBLUE_EX}Лабораторная работа №5 — Метод наименьших квадратов\n")

    stats = compute_stats(x, y)

    print(f"{Fore.GREEN}1. Коэффициент корреляции Пирсона\n")
    print(f"{Fore.WHITE}Среднее X: {stats['mean_x']:.4f}")
    print(f"Среднее Y: {stats['mean_y']:.4f}")
    print(f"σ_X: {stats['std_x']:.4f}")
    print(f"σ_Y: {stats['std_y']:.4f}")
    print(f"Ковариация: {stats['cov_xy']:.4f}")

    print(f"\nКоэффициент корреляции r = {Fore.CYAN}{stats['r_xy']:.4f}{Fore.RESET}")

    strength, direction = interpret_correlation(stats["r_xy"])
    print(f"Интерпретация: {strength} {direction} связь\n")

    print(f"{Fore.GREEN}2. Линейная модель Y(X)\n")
    a, b = compute_regression_params(stats)
    print(f"{Fore.WHITE}a = {a:.4f}")
    print(f"b = {b:.4f}")
    print(f"Модель: {Fore.CYAN}Y = {a:.4f} + {b:.4f} * X{Fore.RESET}\n")

    print(f"{Fore.GREEN}3. Коэффициент детерминации R²\n")
    r2 = compute_r2(stats["r_xy"])
    print(f"{Fore.WHITE}R² = {Fore.CYAN}{r2:.4f}{Fore.RESET}")

    print("\nКачество модели:", end=" ")
    if r2 < 0.3:
        print("низкое")
    elif r2 < 0.5:
        print("удовлетворительное")
    elif r2 < 0.7:
        print("хорошее")
    else:
        print("очень хорошее")

    print(f"\n{Fore.GREEN}4. Построение графика\n")
    plot_regression(x, y, a, b)


if __name__ == "__main__":
    init(autoreset=True)
    main()