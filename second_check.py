import srhimages
import datetime
import pandas as pd
import os
import numpy as np
import srhcp  
from typing import List, Dict, Optional, Union

def check_single_day(date: datetime.date, 
                     start_hour: int = 0, 
                     end_hour: int = 10) -> Dict[str, str]:
    gratings = ['SRH0612', 'SRH1224', 'SRH0306']
    t1 = datetime.datetime.combine(date, datetime.time(start_hour, 0, 0))
    t2 = datetime.datetime.combine(date, datetime.time(end_hour, 0, 0))
    
    try:
        frequencies = srhimages.get_frequencies(t1, t2)
        result = {}
        for grating in gratings:
            result[grating] = '+' if (grating in frequencies and len(frequencies[grating]) > 0) else '-'
        return result
    except Exception as e:
        print(f"Ошибка для даты {date}: {e}")
        return {grating: '-' for grating in gratings}

def sort_data_by_date(data: Dict[datetime.date, Dict[str, str]]) -> Dict[datetime.date, Dict[str, str]]:
    sorted_data = {date: data[date] for date in sorted(data.keys())}
    return sorted_data

def save_to_csv(data: Dict[datetime.date, Dict[str, str]], 
                filename: str = 'data_check.csv'):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Date'
    df = df.sort_index()
    df.to_csv(filename)
    print(f"Результаты сохранены в {filename}")

def load_from_csv(filename: str) -> Dict[datetime.date, Dict[str, str]]:
    if not os.path.exists(filename):
        return {}
    
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.set_index('Date')
    df = df.sort_index()
    result = {}
    for date in df.index:
        result[date] = {col: df.loc[date, col] for col in df.columns}
    return result

def check_data_availability(start_date: Union[datetime.date, datetime.datetime],
                           end_date: Union[datetime.date, datetime.datetime],
                           start_hour: int = 0,
                           end_hour: int = 10,
                           save_to_file: bool = True,
                           load_existing: bool = True) -> Dict[datetime.date, Dict[str, str]]:

    current = start_date.date() if isinstance(start_date, datetime.datetime) else start_date
    end = end_date.date() if isinstance(end_date, datetime.datetime) else end_date
    
    dates = []
    while current <= end:
        dates.append(current)
        current += datetime.timedelta(days=1)
    
    results = {}
    if load_existing and os.path.exists('data_check.csv'):
        results = load_from_csv('data_check.csv')
        print(f"Загружено {len(results)} дней из существующего файла")

    new_days_checked = 0
    for date in dates:
        if date in results:
            print(f"Дата {date} уже проверена, пропускаем")
            continue
        
        print(f"Проверяем {date}...")
        results[date] = check_single_day(date, start_hour, end_hour)
        new_days_checked += 1
    
    if save_to_file and new_days_checked > 0:
        save_to_csv(results, 'data_check.csv')
        print(f"Добавлено {new_days_checked} новых дней")
    
    return results


def check_quality_for_day(date: datetime.date, array: str, frequency: int = 6000) -> str:

    try:
        corr = srhcp.SRHCorrPlot(date, array, frequency, "corrplot_cache")
        
        if corr.data is None:
            return 'NO_DATA'
        
        morning_indices = [i for i, t in enumerate(corr.times) if t.hour < 9]
        
        if len(morning_indices) < 50:
            return 'NO_DATA'
        
        I_vals = corr.I[morning_indices]
        V_vals = corr.V[morning_indices]
        flux_I = corr.flux_I[morning_indices]
        flux_V = corr.flux_V[morning_indices]
        
        I_vals = np.nan_to_num(I_vals, nan=0.0)
        V_vals = np.nan_to_num(V_vals, nan=0.0)
        flux_I = np.nan_to_num(flux_I, nan=0.0)
        flux_V = np.nan_to_num(flux_V, nan=0.0)
        
        if np.max(flux_I) > 1e6 or np.max(np.abs(flux_V)) > 1e6:
            return 'BAD'
        
        def has_big_jump(data):
            if len(data) < 10:
                return False
            diffs = np.abs(np.diff(data))
            positive_diffs = diffs[diffs > 0]
            median_diff = np.median(positive_diffs) if len(positive_diffs) > 0 else 1
            if median_diff == 0:
                median_diff = 1
            max_ratio = np.max(diffs) / median_diff
            return max_ratio > 1500
        
        if has_big_jump(I_vals) or has_big_jump(V_vals):
            return 'PROBLEM'
        
        return 'GOOD'
        
    except Exception as e:
        print(f"  Ошибка при проверке качества {date} {array}: {e}")
        return 'NO_DATA'


def add_quality_check(data: Dict[datetime.date, Dict[str, str]]) -> Dict[datetime.date, Dict[str, str]]:

    gratings = ['SRH0612', 'SRH1224', 'SRH0306']
    new_data = {}
    
    for date, status_dict in data.items():
        new_status_dict = {}
        
        for grating in gratings:
            if status_dict.get(grating) == '+':
                quality = check_quality_for_day(date, grating)
                new_status_dict[grating] = quality
            else:
                new_status_dict[grating] = '-'
        
        new_data[date] = new_status_dict
    
    return new_data


def save_quality_to_csv(data: Dict[datetime.date, Dict[str, str]], 
                        filename: str = 'data_quality.csv'):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Date'
    df.to_csv(filename)
    print(f"Результаты проверки качества сохранены в {filename}")


if __name__ == "__main__":
    
    print("="*70)
    print("ШАГ 1: ПРОВЕРКА НАЛИЧИЯ ДАННЫХ")
    print("="*70)
    
    start = datetime.date(2023, 4, 1)
    end = datetime.date(2023, 4, 10)
    
    availability_results = check_data_availability(start, end, 
                                                    start_hour=0, 
                                                    end_hour=10,
                                                    save_to_file=True,
                                                    load_existing=True)
    
    print("\n" + "="*70)
    print("ШАГ 2: ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("="*70)
    print("(Проверяем только дни, где есть данные '+')")
    
    quality_results = add_quality_check(availability_results)
    
    save_quality_to_csv(quality_results, 'data_quality.csv')
    
    
    print("\n" + "="*70)
    print("СТАТИСТИКА")
    print("="*70)
    
    stats = {'GOOD': 0, 'PROBLEM': 0, 'BAD': 0, '-': 0}
    for status_dict in quality_results.values():
        for status in status_dict.values():
            if status in stats:
                stats[status] += 1
    
    total = sum(stats.values())
    
    print(f"\nВсего проверок: {total}")
    print(f"✅ Хорошие дни (GOOD): {stats['GOOD']} ({stats['GOOD']/total*100:.1f}%)")
    print(f"⚠️ Проблемные дни (PROBLEM - скачки): {stats['PROBLEM']} ({stats['PROBLEM']/total*100:.1f}%)")
    print(f"❌ Плохие дни (BAD - выбросы): {stats['BAD']} ({stats['BAD']/total*100:.1f}%)")
    print(f"❌ Нет данных (-): {stats['-']} ({stats['-']/total*100:.1f}%)")