import pandas as pd
import numpy as np
import re
from datetime import datetime, time

def parse_observation_log_v2(file_path):
    """
    Парсинг первого листа с учетом почасовых температур и погоды
    """
    
    # Загружаем первый лист
    df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
    
    print(f"Загружено строк: {len(df_raw)}")
    print(f"Загружено столбцов: {len(df_raw.columns)}")
    
    # === 1. Анализируем заголовки ===
    header_row1 = df_raw.iloc[0, :].fillna('').astype(str)
    header_row2 = df_raw.iloc[1, :].fillna('').astype(str)
    header_row3 = df_raw.iloc[2, :].fillna('').astype(str)
    
    # Словарь для хранения информации о колонках
    columns_info = {}
    
    # Переменные для отслеживания текущего частотного диапазона
    current_freq_band = None
    
    for col_idx in range(len(df_raw.columns)):
        h1 = header_row1.iloc[col_idx]
        h2 = header_row2.iloc[col_idx]
        h3 = header_row3.iloc[col_idx]
        
        # === Определяем тип колонки ===
        
        # 1. Основная дата (первая колонка)
        if col_idx == 0 or '00:00:00' in str(h3):
            columns_info[col_idx] = {
                'type': 'main_date',
                'name': 'date',
                'hour': None
            }
        
        # 2. Почасовые температуры (формат: "00:00:00", "01:00:00" и т.д.)
        elif re.match(r'\d{2}:\d{2}:\d{2}', str(h3)) and 'температура' in str(h1).lower():
            hour = h3[:2]  # Берем час
            columns_info[col_idx] = {
                'type': 'temperature',
                'name': f'temperature_{hour}:00',
                'hour': hour
            }
        
        # 3. Почасовая погода (идет сразу после температуры)
        elif re.match(r'\d{2}:\d{2}:\d{2}', str(h3)) and 'погода' in str(h2).lower():
            hour = h3[:2]
            columns_info[col_idx] = {
                'type': 'weather',
                'name': f'weather_{hour}:00',
                'hour': hour
            }
        
        # 4. Частотные диапазоны - Пуск
        elif 'Пуск' in str(h2) and ('ГГц' in str(h1) or 'ГГц' in str(h2)):
            # Определяем диапазон
            band = extract_freq_band(h1, h2)
            columns_info[col_idx] = {
                'type': 'start',
                'name': f'{band}_start',
                'band': band
            }
        
        # 5. Частотные диапазоны - Стоп
        elif 'Стоп' in str(h2) and ('ГГц' in str(h1) or 'ГГц' in str(h2)):
            band = extract_freq_band(h1, h2)
            columns_info[col_idx] = {
                'type': 'stop',
                'name': f'{band}_stop',
                'band': band
            }
        
        # 6. Примечания к частотным диапазонам
        elif 'Примечание' in str(h2) and ('ГГц' in str(h1) or 'ГГц' in str(h2)):
            band = extract_freq_band(h1, h2)
            columns_info[col_idx] = {
                'type': 'band_note',
                'name': f'{band}_note',
                'band': band
            }
        
        
        # 8. Дежурные
        elif 'Дежурные' in str(h1):
            columns_info[col_idx] = {
                'type': 'duty',
                'name': 'duty_officer',
                'hour': None
            }
        
        # 9. Пропускаем пустые
        elif h1 == '' and h2 == '' and h3 == '':
            columns_info[col_idx] = {'type': 'skip', 'name': 'skip'}
        
        else:
            # Неизвестный тип, но возможно важный
            columns_info[col_idx] = {
                'type': 'unknown',
                'name': f'unknown_{col_idx}',
                'raw_h1': h1[:50],
                'raw_h2': h2[:50],
                'raw_h3': h3[:50]
            }
    
    # Выводим найденные колонки для отладки
    print("\n=== Найденные типы колонок ===")
    type_counts = {}
    for col_idx, info in columns_info.items():
        col_type = info['type']
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    for col_type, count in sorted(type_counts.items()):
        print(f"  {col_type}: {count} колонок")
    
    # === 2. Собираем данные ===
    data_start = 3
    data_rows = []
    
    for row_idx in range(data_start, len(df_raw)):
        row_data = {}
        has_data = False
        
        for col_idx, info in columns_info.items():
            if info['type'] == 'skip':
                continue
            
            value = df_raw.iloc[row_idx, col_idx]
            
            # Пропускаем пустые значения для неосновных колонок
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
                if info['type'] in ['main_date']:
                    # Для даты пропускаем всю строку
                    has_data = False
                    break
                continue
            
            has_data = True
            
            # Обработка в зависимости от типа
            if info['type'] == 'main_date':
                # Преобразуем дату
                if isinstance(value, datetime):
                    row_data['datetime'] = value
                    row_data['date'] = value.date()
                else:
                    try:
                        row_data['datetime'] = pd.to_datetime(value)
                        row_data['date'] = row_data['datetime'].date()
                    except:
                        row_data['date'] = value
            
            elif info['type'] == 'temperature':
                # Извлекаем число из строки типа "-16С" или "-16.0"
                if isinstance(value, (int, float)):
                    row_data[info['name']] = float(value)
                else:
                    val_str = str(value).strip()
                    # Убираем буквы и пробелы
                    match = re.search(r'(-?\d+\.?\d*)', val_str)
                    if match:
                        row_data[info['name']] = float(match.group(1))
                    else:
                        row_data[info['name']] = np.nan
            
            elif info['type'] == 'weather':
                row_data[info['name']] = str(value).strip()
            
            elif info['type'] in ['start', 'stop', 'band_note', 'general_note']:
                row_data[info['name']] = str(value).strip()
            
            elif info['type'] == 'duty':
                row_data[info['name']] = str(value).strip()
            
            elif info['type'] == 'unknown':
                row_data[info['name']] = str(value).strip()
        
        if has_data and 'datetime' in row_data:
            data_rows.append(row_data)
    
    # === 3. Создаем DataFrame ===
    df = pd.DataFrame(data_rows)
    
    # === 4. Добавляем колонки с временем в минутах для start/stop ===
    def parse_time_to_minutes(time_str):
        if pd.isna(time_str) or time_str == '' or time_str == 'nan':
            return np.nan
        time_str = str(time_str).strip()
        
        # Если это дата, берем только время
        if '-' in time_str and len(time_str) > 10:
            parts = time_str.split()
            if len(parts) >= 2:
                time_str = parts[1]
        
        # Формат "01 00"
        if ' ' in time_str:
            parts = time_str.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]) * 60 + int(parts[1])
        
        # Формат "01:00"
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]) * 60 + int(parts[1])
        
        # Формат "0145"
        if time_str.isdigit() and len(time_str) >= 3:
            if len(time_str) == 3:
                return int(time_str[0]) * 60 + int(time_str[1:3])
            elif len(time_str) == 4:
                return int(time_str[:2]) * 60 + int(time_str[2:4])
        
        return np.nan
    
    # Добавляем минуты для всех start/stop колонок
    time_columns = [col for col in df.columns if '_start' in col or '_stop' in col]
    for col in time_columns:
        df[f'{col}_min'] = df[col].apply(parse_time_to_minutes)
    
    # === 5. Создаем "длинный" формат для почасовых данных ===
    # Это позволит удобнее анализировать температуру и погоду по часам
    
    # Находим все колонки с температурой и погодой
    temp_cols = [col for col in df.columns if col.startswith('temperature_')]
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    
    # Создаем список для длинного формата
    long_format_rows = []
    
    for idx, row in df.iterrows():
        date = row.get('datetime', None)
        if pd.isna(date):
            continue
        
        # Для каждого часа
        for hour in range(0, 24):
            hour_str = f"{hour:02d}:00"
            temp_col = f'temperature_{hour_str}'
            weather_col = f'weather_{hour_str}'
            
            temp = row.get(temp_col, np.nan)
            weather = row.get(weather_col, '')
            
            # Добавляем только если есть данные
            if not pd.isna(temp) or (isinstance(weather, str) and weather.strip() and weather != 'nan'):
                long_format_rows.append({
                    'datetime': date,
                    'date': date.date() if hasattr(date, 'date') else date,
                    'hour': hour,
                    'temperature': temp if not pd.isna(temp) else np.nan,
                    'weather': weather if isinstance(weather, str) and weather.strip() and weather != 'nan' else '',
                    # Копируем основные поля из исходной строки
                    'general_note': row.get('general_note', ''),
                    'duty_officer': row.get('duty_officer', '')
                })
    
    df_long = pd.DataFrame(long_format_rows)
    
    return df, df_long


def extract_freq_band(h1, h2):
    """Извлечение названия частотного диапазона"""
    full_text = str(h1) + ' ' + str(h2)
    
    if '4-8' in full_text:
        return '4-8_GHz'
    elif '0,05-3' in full_text or '0.05-3' in full_text or 'Callisto' in full_text:
        return '005-3_GHz'
    elif '3-24' in full_text:
        return '3-24_GHz'
    elif '3-6' in full_text:
        return '3-6_GHz'
    elif '6-12' in full_text:
        return '6-12_GHz'
    elif '12-24' in full_text:
        return '12-24_GHz'
    elif '2-24' in full_text:
        return '2-24_GHz'
    else:
        # Очищаем от лишних символов
        clean = re.sub(r'[^\w\-]', '_', full_text)
        return clean[:30]


def print_summary(df, df_long):
    """Вывод статистики"""
    print("\n" + "="*60)
    print("СТАТИСТИКА")
    print("="*60)
    
    print(f"\n📊 Широкий формат (исходный):")
    print(f"  - Записей: {len(df)}")
    print(f"  - Колонок: {len(df.columns)}")
    
    print(f"\n📊 Длинный формат (почасовой):")
    print(f"  - Записей: {len(df_long)}")
    if len(df_long) > 0:
        print(f"  - Период: с {df_long['datetime'].min()} по {df_long['datetime'].max()}")
        
        # Статистика по температуре
        valid_temp = df_long['temperature'].dropna()
        if len(valid_temp) > 0:
            print(f"  - Температура: мин={valid_temp.min():.1f}°C, макс={valid_temp.max():.1f}°C, сред={valid_temp.mean():.1f}°C")
        
        # Статистика по погоде
        weather_counts = df_long['weather'].value_counts().head(10)
        print(f"\n  🌤 Топ-10 погодных условий:")
        for weather, count in weather_counts.items():
            if weather and weather != '':
                print(f"     - {weather}: {count} раз(а)")
    
    # Статистика по частотным диапазонам
    print(f"\n📡 Частотные диапазоны:")
    for col in df.columns:
        if col.endswith('_start'):
            band = col.replace('_start', '')
            active = df[col].notna().sum()
            if active > 0:
                print(f"  - {band}: {active} наблюдений")


def save_parsed_data(df, df_long, prefix="parsed"):
    """Сохранение данных"""
    # Сохраняем широкий формат
    df.to_csv(f"{prefix}_wide.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Сохранен широкий формат: {prefix}_wide.csv")
    
    # Сохраняем длинный формат (почасовой)
    df_long.to_csv(f"{prefix}_long.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Сохранен длинный формат: {prefix}_long.csv")


# === Использование ===
if __name__ == "__main__":
    file_path = "Radioheliograph.xlsx"
    
    try:
        # Парсим
        df_wide, df_long = parse_observation_log_v2(file_path)
        
        # Выводим статистику
        print_summary(df_wide, df_long)
        
        # Показываем пример длинного формата
        print("\n=== Пример длинного формата (почасовой) ===")
        if len(df_long) > 0:
            print(df_long.head(10).to_string())
        
        # Сохраняем
        save_parsed_data(df_wide, df_long)
        
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()