import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import calendar


def load_all_data(data_dir: str) -> dict:
    all_data = {}
    
    if not os.path.exists(data_dir):
        print(f" Папка не найдена: {data_dir}")
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
    time_range = grating_data.get("time_range", "NO_DATA")
    flux_data = grating_data.get("flux", {})
    
    status_counts = defaultdict(int)
    all_freqs_info = []
    
    for freq_str, freq_info in flux_data.items():
        state = freq_info.get("state", "NO_DATA")
        status_counts[state] += 1
        
        all_freqs_info.append({
            "frequency": freq_str,
            "state": state,
            "comment": freq_info.get("comment", ""),
            "flux_median": freq_info.get("flux_I_median", 0),
            "flux_mean": freq_info.get("flux_I_mean", 0),
            "sfu_ratio": freq_info.get("sfu_ratio", 0),
            "expected_sfu": freq_info.get("expected_sfu", 0),
            "time_start": freq_info.get("time_start", ""),
            "time_range": freq_info.get("time_range", "")
        })
    
    all_freqs_info.sort(key=lambda x: int(x["frequency"]))
    
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
        "all_freqs": all_freqs_info,
        "total_freqs": len(flux_data),
        "availability": grating_data.get("availability", False),
        "journal_notes": grating_data.get("journal_notes", {}).get("details", "")
    }


def create_calendar_heatmap(all_data: dict, grating: str, year: int = None, save_path: str = None):
    if not all_data:
        print(" Нет данных для визуализации")
        return None, None, None
    
    if year is None:
        dates = list(all_data.keys())
        if dates:
            year = max(d.year for d in dates)
        else:
            year = datetime.now().year
    
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    days_data = {}
    current_date = start_date
    while current_date <= end_date:
        if current_date in all_data and grating in all_data[current_date]:
            analysis = analyze_grating_status(all_data[current_date][grating])
            days_data[current_date] = analysis
        current_date += timedelta(days=1)
    
    last_day = datetime(year, 12, 31).date()
    weeks_in_year = last_day.isocalendar()[1]
    if weeks_in_year == 1:
        weeks_in_year = 52
    
    fig = plt.figure(figsize=(max(16, weeks_in_year * 0.35), 8))
    ax = plt.subplot(111)
    
    first_day = datetime(year, 1, 1).date()
    current_date = first_day
    
    while current_date.year == year:
        weekday = current_date.weekday()
        week = current_date.isocalendar()[1]
        
        if current_date.month == 12 and week < 10:
            week += 52
        
        col = week - 1
        row = weekday
        
        if 0 <= col < weeks_in_year:
            if current_date in days_data:
                analysis = days_data[current_date]
                
                if analysis["overall_status"] == "BAD":
                    color = '#ff4444'
                    bad_count = analysis["status_counts"].get("BAD", 0)
                    problem_count = analysis["status_counts"].get("PROBLEM", 0)
                    if bad_count > 0 and problem_count > 0:
                        label = f"B:{bad_count}/P:{problem_count}"
                    elif bad_count > 0:
                        label = f"B:{bad_count}"
                    else:
                        label = ""
                elif analysis["overall_status"] == "PROBLEM":
                    color = '#ffaa00'
                    problem_count = analysis["status_counts"].get("PROBLEM", 0)
                    label = f"P:{problem_count}" if problem_count > 0 else ""
                elif analysis["overall_status"] == "GOOD":
                    color = '#44aa44'
                    label = ""
                else:
                    color = '#eeeeee'
                    label = ""
                
                has_notes = bool(analysis.get("journal_notes", ""))
                if has_notes:
                    label += " 📝" if label else "📝"
            else:
                color = '#f0f0f0'
                label = ""
            
            rect = Rectangle(
                (col + 0.05, row + 0.05),
                0.9, 0.9,
                linewidth=0.5,
                edgecolor='white',
                facecolor=color,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            if label:
                fontsize = 7 if len(label) <= 4 else 6 if len(label) <= 8 else 5
                ax.text(
                    col + 0.5, row + 0.5,
                    label,
                    ha='center', va='center',
                    fontsize=fontsize,
                    color='white' if color in ['#ff4444', '#44aa44'] else 'black',
                    fontweight='bold'
                )
            
            rect.set_picker(current_date in days_data)
            rect.cell_data = {
                'date': current_date,
                'has_data': current_date in days_data
            }
        
        current_date += timedelta(days=1)
    
    ax.set_xlim(-0.5, weeks_in_year - 0.5)
    ax.set_ylim(0, 7)
    
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    month_positions = []
    
    for month in range(1, 13):
        first_of_month = datetime(year, month, 1).date()
        week_of_month = first_of_month.isocalendar()[1] - 1
        if week_of_month < weeks_in_year:
            month_positions.append(week_of_month)
    
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_names[:len(month_positions)])
    
    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    ax.set_yticklabels(day_names)
    ax.invert_yaxis()
    
    good_days = sum(1 for d in days_data.values() if d["overall_status"] == "GOOD")
    problem_days = sum(1 for d in days_data.values() if d["overall_status"] == "PROBLEM")
    bad_days = sum(1 for d in days_data.values() if d["overall_status"] == "BAD")
    total_days = len(days_data)
    
    title = f'{grating} - Качество данных за {year} год'
    stats_text = f'Всего дней с данными: {total_days} | GOOD: {good_days} | PROBLEM: {problem_days} | BAD: {bad_days}'
    
    plt.title(f'{title}\n{stats_text}', fontsize=14, fontweight='bold', pad=15)
    
    legend_elements = [
        mpatches.Patch(color='#44aa44', label='GOOD'),
        mpatches.Patch(color='#ffaa00', label='PROBLEM'),
        mpatches.Patch(color='#ff4444', label='BAD'),
        mpatches.Patch(color='#f0f0f0', label='Нет данных')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
    
    plt.figtext(0.5, 0.01, 'Кликните по квадрату для создания PDF отчета', 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    
    return fig, ax, days_data


def generate_day_pdf_matplotlib(date: datetime.date, day_data: dict, grating: str, output_dir: str = "pdf_reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"{grating}_{date.isoformat()}.pdf")
    
    analysis = analyze_grating_status(day_data)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    with PdfPages(filename) as pdf:
        
        def wrap_text(text, max_chars_per_line=80):
            """Разбивает текст на строки с учетом слов"""
            if not text:
                return []
            
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if len(test_line) <= max_chars_per_line:
                    current_line = test_line
                else:
                    if len(word) > max_chars_per_line:
                        if current_line:
                            lines.append(current_line)
    
                        for i in range(0, len(word), max_chars_per_line):
                            lines.append(word[i:i+max_chars_per_line])
                        current_line = ""
                    else:
                        lines.append(current_line)
                        current_line = word
            
            if current_line:
                lines.append(current_line)
            
            return lines
 
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        y_pos = 0.95
        
        ax.text(0.05, y_pos, 'Отчет о качестве данных', fontsize=18, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.05
        ax.text(0.05, y_pos, f'Дата: {date.isoformat()}    Решетка: {grating}', fontsize=14, transform=ax.transAxes)
        
        y_pos -= 0.06
        
        ax.text(0.05, y_pos, 'Общая информация:', fontsize=14, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.03
        
        status_color = 'green' if analysis['overall_status'] == 'GOOD' else 'orange' if analysis['overall_status'] == 'PROBLEM' else 'red'
        
        info_lines = [
            f"Общий статус: {analysis['overall_status']}",
            f"Доступность: {'Да' if analysis['availability'] else 'Нет'}",
            f"Time Range: {analysis['time_range']}",
            f"Всего частот: {analysis['total_freqs']}",
            f"GOOD: {analysis['status_counts'].get('GOOD', 0)} | PROBLEM: {analysis['status_counts'].get('PROBLEM', 0)} | BAD: {analysis['status_counts'].get('BAD', 0)}"
        ]
        
        for line in info_lines:
            ax.text(0.08, y_pos, line, fontsize=11, transform=ax.transAxes)
            y_pos -= 0.025
        
        y_pos -= 0.01
        
        if analysis["journal_notes"]:
          
            note_lines = wrap_text(analysis["journal_notes"], max_chars_per_line=80)
            
            if note_lines:
                
                notes_height = len(note_lines) * 0.022 + 0.06
                
                if y_pos - notes_height < 0.1:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(12, 10))
                    ax.axis('off')
                    y_pos = 0.95
                
                ax.text(0.05, y_pos, 'Заметки из журнала ошибок:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.03
                
                for line in note_lines:
                    if y_pos < 0.1:
                        pdf.savefig(fig)
                        plt.close(fig)
                        fig, ax = plt.subplots(figsize=(12, 10))
                        ax.axis('off')
                        y_pos = 0.95
                    
                    ax.text(0.08, y_pos, line, fontsize=10, transform=ax.transAxes,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
                    y_pos -= 0.025
                
                y_pos -= 0.02
        
        ax.text(0.05, y_pos, 'Данные по всем частотам:', fontsize=14, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.03
        
        col_labels = ['Частота', 'Статус', 'Медиана', 'SFU', 'Отнош.', 'Комментарий']
        col_widths = [0.08, 0.08, 0.08, 0.08, 0.08, 0.4]
        
        x_pos = 0.05
        for i, (label, width) in enumerate(zip(col_labels, col_widths)):
            ax.text(x_pos, y_pos, label, fontsize=9, fontweight='bold', color='white',
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='#666666', alpha=0.8))
            x_pos += width
        
        y_pos -= 0.025
        
        for row_idx, freq_info in enumerate(analysis["all_freqs"]):
            if y_pos < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.axis('off')
                y_pos = 0.95
                
                x_pos = 0.05
                for i, (label, width) in enumerate(zip(col_labels, col_widths)):
                    ax.text(x_pos, y_pos, label, fontsize=9, fontweight='bold', color='white',
                           transform=ax.transAxes,
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='#666666', alpha=0.8))
                    x_pos += width
                y_pos -= 0.025
            
            comment = freq_info["comment"]
            if len(comment) > 70:
                comment = comment[:67] + "..."
            
            if freq_info["state"] == "BAD":
                bg_color = '#ffe6e6'
            elif freq_info["state"] == "PROBLEM":
                bg_color = '#fff3e6'
            else:
                bg_color = 'white'
            
            row_data = [
                freq_info["frequency"],
                freq_info["state"],
                f"{freq_info['flux_median']:.1f}",
                f"{freq_info['expected_sfu']:.1f}",
                f"{freq_info['sfu_ratio']:.2f}",
                comment
            ]
            
            x_pos = 0.05
            for i, (value, width) in enumerate(zip(row_data, col_widths)):
                ax.text(x_pos, y_pos, value, fontsize=8, transform=ax.transAxes,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor=bg_color, alpha=0.5))
                x_pos += width
            
            y_pos -= 0.022
        
        pdf.savefig(fig)
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        y_pos = 0.9
        
        if analysis["total_freqs"] > 0:
            ax.text(0.05, y_pos, 'Распределение статусов:', fontsize=14, fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.05
            
            good_pct = analysis["status_counts"].get("GOOD", 0) / analysis["total_freqs"] * 100
            problem_pct = analysis["status_counts"].get("PROBLEM", 0) / analysis["total_freqs"] * 100
            bad_pct = analysis["status_counts"].get("BAD", 0) / analysis["total_freqs"] * 100
            
            stats = [
                f"GOOD: {analysis['status_counts'].get('GOOD', 0)} частот ({good_pct:.1f}%)",
                f"PROBLEM: {analysis['status_counts'].get('PROBLEM', 0)} частот ({problem_pct:.1f}%)",
                f"BAD: {analysis['status_counts'].get('BAD', 0)} частот ({bad_pct:.1f}%)"
            ]
            
            colors_stats = ['green', 'orange', 'red']
            for stat, color in zip(stats, colors_stats):
                ax.text(0.08, y_pos, stat, fontsize=12, color=color, transform=ax.transAxes)
                y_pos -= 0.04
            
            categories = ['GOOD', 'PROBLEM', 'BAD']
            values = [
                analysis["status_counts"].get("GOOD", 0),
                analysis["status_counts"].get("PROBLEM", 0),
                analysis["status_counts"].get("BAD", 0)
            ]
            bar_colors = ['green', 'orange', 'red']
            
            ax_bar = fig.add_axes([0.15, 0.3, 0.7, 0.3])
            bars = ax_bar.bar(categories, values, color=bar_colors, alpha=0.7)
            ax_bar.set_ylabel('Количество частот')
            ax_bar.set_title('Распределение статусов частот')
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              str(value), ha='center', va='bottom')
        
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"PDF отчет создан: {filename}")
    return filename

def on_click(event, fig, ax, days_data, grating, output_dir="pdf_reports"):
    if hasattr(event, 'artist') and hasattr(event.artist, 'cell_data'):
        cell = event.artist.cell_data
        
        if cell.get('has_data', False):
            date = cell['date']
            print(f"\nГенерация PDF отчета для {date.isoformat()} / {grating}...")
            
            for loaded_date, day_data in all_data_cache.items():
                if loaded_date == date and grating in day_data:
                    generate_day_pdf_matplotlib(date, day_data[grating], grating, output_dir)
                    break


all_data_cache = {}


def create_interactive_calendar(data_dir: str, grating: str, year: int = None):
    global all_data_cache
    
    print(f"\nЗагрузка данных для {grating}...")
    all_data_cache = load_all_data(data_dir)
    
    if not all_data_cache:
        print("Нет данных")
        return
    
    fig, ax, days_data = create_calendar_heatmap(all_data_cache, grating, year)
    
    if fig is not None:
        fig.canvas.mpl_connect(
            'pick_event',
            lambda event: on_click(event, fig, ax, days_data, grating)
        )
        plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ИНТЕРАКТИВНАЯ КАЛЕНДАРНАЯ ВИЗУАЛИЗАЦИЯ")
    print("="*70)
    
    data_dir = "data_quality_files"
    
    if not os.path.exists(data_dir):
        print(f"\nПапка с данными не найдена: {data_dir}")
        exit(1)
    
    for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
        print(f"\n{'='*70}")
        print(f"ЗАПУСК КАЛЕНДАРЯ ДЛЯ {grating}")
        print("="*70)
        print("Кликайте по квадратам для создания PDF отчетов")
        print("   Закройте окно для перехода к следующей решетке\n")
        
        create_interactive_calendar(data_dir, grating, year=2024)
    
    print("\nГотово! PDF отчеты сохранены в папку 'pdf_reports'")