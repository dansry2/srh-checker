import pandas as pd
import re

def extract_temp_from_date(val_str):
    """
    Извлекает температуру из даты, где день = температура (со знаком минус если нужно).
    Пример: '2023-10-09 00:00:00' -> 9.0
    """
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', val_str)
    if match:
        day = int(match.group(3))
        return float(day)
    return None

def fix_time_value(val):
    """Нормализует формат времени к HH:MM"""
    if pd.isna(val) or str(val).strip() == '':
        return ''
    
    val_str = str(val).strip()
    
    # 0. Формат с секундами "HH:MM:SS" -> обрезаем до "HH:MM"
    if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str):
        parts = val_str.split(':')
        return f"{int(parts[0]):02d}:{parts[1]}"
    
    # 1. Пытаемся извлечь время из даты (испорченные Excel данные)
    time_from_date = extract_time_from_date(val_str)
    if time_from_date:
        return time_from_date
    
    # 2. Правильный формат "HH:MM" - нормализуем
    if re.match(r'^\d{1,2}:\d{2}$', val_str):
        parts = val_str.split(':')
        return f"{int(parts[0]):02d}:{parts[1]}"
    
    # 3. Формат с пробелом "HH MM"
    if re.match(r'^\d{1,2}\s+\d{2}$', val_str):
        parts = val_str.split()
        return f"{int(parts[0]):02d}:{parts[1]}"
    
    # 4. Четыре цифры подряд "0900"
    if re.match(r'^\d{4}$', val_str):
        return f"{val_str[:2]}:{val_str[2:]}"
    
    # 5. Число с точкой "0.45" -> "00:45"
    if '.' in val_str:
        try:
            parts = val_str.split('.')
            if len(parts) == 2:
                hours = int(parts[0]) if parts[0] else 0
                minutes = int(parts[1])
                if minutes < 60:
                    return f"{hours:02d}:{minutes:02d}"
        except:
            pass
    
    # 6. Просто число (например, "9") -> "09:00"
    try:
        num = int(val_str)
        if 0 <= num <= 23:
            return f"{num:02d}:00"
    except:
        pass
    
    return val_str

def extract_time_from_date(val_str):
    """Извлекает время из даты, где день = часы."""
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', val_str)
    if match:
        day = int(match.group(3))
        
        # Проверяем, есть ли в дате минуты
        time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})', val_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            if hour != 0 or minute != 0:
                return f"{hour:02d}:{minute:02d}"
        
        # Если время 00:00:00, используем день как часы
        if 0 <= day <= 23:
            return f"{day:02d}:00"
    
    return None

def fix_temp_value(val):
    """Нормализует температурные значения к формату '16.0' (без 'C')"""
    if pd.isna(val) or str(val).strip() == '':
        return ''
    
    val_str = str(val).strip()
    
    # 1. Проверяем, не дата ли это (артефакт Excel)
    temp_from_date = extract_temp_from_date(val_str)
    if temp_from_date is not None:
        return f"{temp_from_date:.1f}"
    
    # 2. Убираем 'C', 'С', 'c', пробелы
    val_str = val_str.upper().replace('C', '').replace('С', '').strip()
    
    # 3. Пытаемся извлечь число (может быть целое или с точкой)
    # Ищем паттерн: опциональный минус, цифры, опциональная точка и цифры
    match = re.search(r'(-?\d+\.?\d*)', val_str)
    if match:
        num_str = match.group(1)
        try:
            num = float(num_str)
            return f"{num:.1f}"
        except:
            pass
    
    # 4. Если ничего не подошло, возвращаем очищенную строку
    return val_str

# ========== ОСНОВНОЙ БЛОК ==========
print("Загрузка файла...")
df = pd.read_csv('parsed_wide_reorganized.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)

# Находим все временные колонки
time_columns = [col for col in df.columns if '_start' in col or '_stop' in col]
print(f"Найдено временных колонок: {len(time_columns)}")

# Находим все температурные колонки
temp_columns = [col for col in df.columns if 'Температура' in col]
print(f"Найдено температурных колонок: {len(temp_columns)}")

# СОХРАНЯЕМ МАСКИ ИЗНАЧАЛЬНО ПУСТЫХ ЯЧЕЕК
empty_mask_time = {}
for col in time_columns:
    empty_mask_time[col] = df[col].apply(lambda x: str(x).strip() in ['', 'nan', 'NaN', 'None'])

empty_mask_temp = {}
for col in temp_columns:
    empty_mask_temp[col] = df[col].apply(lambda x: str(x).strip() in ['', 'nan', 'NaN', 'None'])

# Применяем нормализацию к времени
for col in time_columns:
    df[col] = df[col].apply(fix_time_value)

# Применяем нормализацию к температурам
for col in temp_columns:
    df[col] = df[col].apply(fix_temp_value)

# ВОССТАНАВЛИВАЕМ ПУСТЫЕ ЗНАЧЕНИЯ
for col in time_columns:
    df.loc[empty_mask_time[col], col] = ''

for col in temp_columns:
    df.loc[empty_mask_temp[col], col] = ''

# Сохраняем результат
output_filename = 'parsed_wide_fixed.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\nГотово! Файл сохранен как: {output_filename}")

# Показываем примеры преобразований температур
print("\nПримеры преобразований температур:")
for col in temp_columns[:3]:
    sample_vals = df[col][df[col] != ''].head(5).tolist()
    if sample_vals:
        print(f"  {col}: {sample_vals}")
    else:
        print(f"  {col}: (все пусто)")

# Статистика
print("\nСтатистика по температурам:")
for col in temp_columns:
    non_empty = df[col][df[col] != '']
    if len(non_empty) > 0:
        unique = non_empty.unique()
        print(f"  {col}: {len(non_empty)} значений, уникальных: {len(unique)}")