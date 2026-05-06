import json
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_all_data(data_dir: str) -> dict:
    """
    Загружает все JSON файлы из папки.
    Возвращает словарь {date: data}
    """
    all_data = {}
    
    if not os.path.exists(data_dir):
        print(f"❌ Папка не найдена: {data_dir}")
        return all_data
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            day_data = json.load(f)
        
        date_str = day_data.get("date", filename.replace('.json', ''))
        date_obj = datetime.fromisoformat(date_str).date()
        
        all_data[date_obj] = day_data
    
    return all_data


def analyze_grating_status(grating_data: dict) -> dict:
    """
    Анализирует статус решетки за день.
    Возвращает общий статус и статистику.
    """
    time_range = grating_data.get("time_range", "NO_DATA")
    flux_data = grating_data.get("flux", {})
    
    # Считаем статусы по частотам
    status_counts = defaultdict(int)
    problem_freqs = []
    bad_freqs = []
    
    for freq_str, freq_info in flux_data.items():
        state = freq_info.get("state", "NO_DATA")
        status_counts[state] += 1
        
        if state == "BAD":
            bad_freqs.append(freq_str)
        elif state == "PROBLEM":
            problem_freqs.append(freq_str)
    
    # Определяем общий статус по логике:
    # 1. Если time_range BAD -> всегда красный
    # 2. Если time_range PROBLEM -> смотрим частоты
    # 3. Если time_range GOOD -> смотрим частоты
    
    if time_range == "BAD":
        overall_status = "BAD"
    elif status_counts.get("BAD", 0) > 0:
        overall_status = "BAD"
    elif status_counts.get("PROBLEM", 0) > 0 or time_range == "PROBLEM":
        overall_status = "PROBLEM"
    else:
        overall_status = "GOOD"
    
    return {
        "overall_status": overall_status,
        "time_range": time_range,
        "status_counts": dict(status_counts),
        "bad_freqs": bad_freqs,
        "problem_freqs": problem_freqs,
        "total_freqs": len(flux_data)
    }


def plot_grating_timeline(all_data: dict, grating: str, save_path: str = None):
    """
    Строит график статусов для конкретной решетки по всем дням.
    """
    # Собираем данные по дням
    dates = []
    statuses = []
    time_ranges = []
    bad_counts = []
    problem_counts = []
    good_counts = []
    journal_notes_flags = []
    
    for date_obj in sorted(all_data.keys()):
        day_data = all_data[date_obj]
        
        if grating not in day_data:
            continue
        
        grating_data = day_data[grating]
        analysis = analyze_grating_status(grating_data)
        
        dates.append(date_obj)
        statuses.append(analysis["overall_status"])
        time_ranges.append(analysis["time_range"])
        
        status_counts = analysis["status_counts"]
        bad_counts.append(status_counts.get("BAD", 0))
        problem_counts.append(status_counts.get("PROBLEM", 0))
        good_counts.append(status_counts.get("GOOD", 0))
        
        # Проверяем наличие journal_notes
        has_notes = "journal_notes" in grating_data and grating_data["journal_notes"].get("details", "")
        journal_notes_flags.append(has_notes)
    
    if not dates:
        print(f"❌ Нет данных для решетки {grating}")
        return
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 3]})
    
    # Верхний график - статусы точками
    colors = []
    for status in statuses:
        if status == "BAD":
            colors.append('red')
        elif status == "PROBLEM":
            colors.append('orange')
        else:
            colors.append('green')
    
    ax1.scatter(dates, [1] * len(dates), c=colors, s=100, zorder=5, edgecolors='black', linewidth=0.5)
    
    # Добавляем метки для дней с journal_notes
    for i, (date, has_notes) in enumerate(zip(dates, journal_notes_flags)):
        if has_notes:
            ax1.annotate('📝', (date, 1), textcoords="offset points", xytext=(0, 15),
                        ha='center', fontsize=12)
    
    # Настройка верхнего графика
    ax1.set_ylim(0.5, 1.5)
    ax1.set_yticks([])
    ax1.set_xlabel('')
    ax1.set_title(f'{grating} - Общий статус по дням', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Легенда для верхнего графика
    legend_elements = [
        mpatches.Patch(color='green', label='GOOD (все частоты OK)'),
        mpatches.Patch(color='orange', label='PROBLEM (есть проблемы)'),
        mpatches.Patch(color='red', label='BAD (критичные проблемы)'),
        mpatches.Patch(color='white', label='📝 Есть записи в журнале')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))
    
    # Нижний график - количество частот по статусам
    x = range(len(dates))
    width = 0.25
    
    bars_good = ax2.bar([i - width for i in x], good_counts, width, label='GOOD', color='green', alpha=0.7)
    bars_problem = ax2.bar(x, problem_counts, width, label='PROBLEM', color='orange', alpha=0.7)
    bars_bad = ax2.bar([i + width for i in x], bad_counts, width, label='BAD', color='red', alpha=0.7)
    
    # Добавляем значения на столбцы
    for bars in [bars_good, bars_problem, bars_bad]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
    
    # Настройка нижнего графика
    ax2.set_xlabel('Дата', fontsize=12)
    ax2.set_ylabel('Количество частот', fontsize=12)
    ax2.set_title(f'Распределение статусов по частотам', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax2.grid(True, alpha=0.3)
    
    # Добавляем аннотацию о time_range
    for i, (date, tr) in enumerate(zip(dates, time_ranges)):
        if tr != "GOOD":
            ax2.annotate(f'TR:{tr}', (i, max(bad_counts[i], problem_counts[i], good_counts[i])),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=7, color='red' if tr == 'BAD' else 'orange')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 График сохранен: {save_path}")
    
    plt.show()


def plot_all_gratings(data_dir: str, save_dir: str = None):
    """
    Строит графики для всех трех решеток.
    """
    print("📊 Загрузка данных...")
    all_data = load_all_data(data_dir)
    
    if not all_data:
        print("❌ Нет данных для визуализации")
        return
    
    print(f"📁 Загружено данных за {len(all_data)} дней")
    
    # Статистика по всем дням
    print("\n📈 Общая статистика:")
    for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
        good_days = 0
        problem_days = 0
        bad_days = 0
        total_days = 0
        
        for date_obj, day_data in all_data.items():
            if grating in day_data:
                total_days += 1
                analysis = analyze_grating_status(day_data[grating])
                
                if analysis["overall_status"] == "GOOD":
                    good_days += 1
                elif analysis["overall_status"] == "PROBLEM":
                    problem_days += 1
                else:
                    bad_days += 1
        
        print(f"  {grating}: {total_days} дней (✅{good_days} ⚠️{problem_days} ❌{bad_days})")
    
    # Строим графики для каждой решетки
    for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
        print(f"\n📊 Построение графика для {grating}...")
        
        save_path = None
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"{grating}_status.png")
        
        plot_grating_timeline(all_data, grating, save_path)


def create_summary_plot(all_data: dict, save_path: str = None):
    """
    Создает сводный график по всем решеткам.
    """
    gratings = ['SRH0306', 'SRH0612', 'SRH1224']
    
    # Подготовка данных
    dates = sorted(all_data.keys())
    
    fig, axes = plt.subplots(len(gratings), 1, figsize=(16, 10))
    
    for idx, grating in enumerate(gratings):
        ax = axes[idx]
        
        statuses = []
        for date in dates:
            if grating in all_data[date]:
                analysis = analyze_grating_status(all_data[date][grating])
                statuses.append(analysis["overall_status"])
            else:
                statuses.append("NO_DATA")
        
        colors = []
        for status in statuses:
            if status == "BAD":
                colors.append('red')
            elif status == "PROBLEM":
                colors.append('orange')
            elif status == "GOOD":
                colors.append('green')
            else:
                colors.append('gray')
        
        ax.scatter(dates, [1] * len(dates), c=colors, s=80, zorder=5, 
                  edgecolors='black', linewidth=0.5)
        
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.set_ylabel(grating, fontsize=12, fontweight='bold', rotation=0, labelpad=40)
        ax.grid(True, axis='x', alpha=0.3)
        
        if idx == 0:
            ax.set_title('Сводный статус по всем решеткам', fontsize=14, fontweight='bold')
    
    # Общая легенда
    legend_elements = [
        mpatches.Patch(color='green', label='GOOD'),
        mpatches.Patch(color='orange', label='PROBLEM'),
        mpatches.Patch(color='red', label='BAD'),
        mpatches.Patch(color='gray', label='NO DATA')
    ]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Сводный график сохранен: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ВИЗУАЛИЗАЦИЯ КАЧЕСТВА ДАННЫХ")
    print("="*70)
    
    data_dir = "data_quality_files"
    plots_dir = "quality_plots"
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Папка с данными не найдена: {data_dir}")
        print("Сначала запустите проверку качества данных")
        exit(1)
    
    # Строим графики для каждой решетки отдельно
    plot_all_gratings(data_dir, plots_dir)
    
    # Строим сводный график
    print(f"\n📊 Создание сводного графика...")
    all_data = load_all_data(data_dir)
    if all_data:
        summary_path = os.path.join(plots_dir, "summary_all_gratings.png") if plots_dir else None
        create_summary_plot(all_data, summary_path)
    
    print("\n" + "="*70)
    print("✅ Визуализация завершена!")
    print(f"📁 Графики сохранены в папку: {plots_dir}")
    print("="*70)