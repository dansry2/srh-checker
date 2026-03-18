import srhimages
import datetime
import pandas as pd
import os
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

def save_to_csv(data: Dict[datetime.date, Dict[str, str]], 
                filename: str = 'data_check.csv'):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Date'
    df.to_csv(filename)
    print(f"Результаты сохранены в {filename}")

def load_from_csv(filename: str) -> Dict[datetime.date, Dict[str, str]]:
    if not os.path.exists(filename):
        return {}
    
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.set_index('Date')
    
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
    
start = datetime.date(2022, 4, 1)
end = datetime.date(2022, 6, 1)
results = check_data_availability(start, end, save_to_file=False)

for date, status in results.items():
    print(f"{date}: {status}")

results_all = check_data_availability(
    datetime.date(2022, 4, 1), 
    datetime.date(2022, 6, 10),
    save_to_file=True
)
