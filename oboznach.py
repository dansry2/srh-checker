import pandas as pd

def reorganize_csv(input_file, output_file, column_mapping=None, column_order=None):
    """
    Переименование и перестановка колонок в уже сохраненном CSV
    """
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"Исходные колонки: {list(df.columns)}")
    df = df.drop(columns=['4-8_GHz_start_min'])
    # Переименовываем колонки
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"После переименования: {list(df.columns)}")
    
    # Меняем порядок колонок
    if column_order:
        # Берем только те колонки, которые существуют
        ordered_cols = [col for col in column_order if col in df.columns]
        # Добавляем остальные колонки в конец
        other_cols = [col for col in df.columns if col not in ordered_cols]
        df = df[ordered_cols + other_cols]
        print(f"Новый порядок: {list(df.columns)}")
    
    # Сохраняем
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Сохранено в {output_file}")
    
    return df

# Пример использования
if __name__ == "__main__":
    # Для широкого формата
    wide_mapping = {
        'datetime': 'Дата_время',
        'Температура': 'Температура',
        'duty_officer': 'Дежурные',
        'date':'Дата',
        'unknown_1':'Температура в 00:00',
        'unknown_2':'Погода в 00:00',
        'unknown_22':'Погода в 10:00',
        'unknown_3':'Температура в 01:00',
        'unknown_4':'Погода в 01:00',
        'unknown_5':'Температура в 02:00',
        'unknown_6':'Погода в 02:00',
        'unknown_7':'Температура в 03:00',
        'unknown_8':'Погода в 03:00',
        'unknown_9':'Температура в 04:00',
        'unknown_10':'Погода в 04:00',
        'unknown_11':'Температура в 05:00',
        'unknown_12':'Погода в 05:00',
        'unknown_13':'Температура в 06:00',
        'unknown_14':'Погода в 06:00',
        'unknown_15':'Температура в 07:00',
        'unknown_16':'Погода в 07:00',
        'unknown_17':'Температура в 08:00',
        'unknown_18':'Погода в 08:00',
        'unknown_19':'Температура в 09:00',
        'unknown_20':'Погода в 09:00',
        'unknown_21':'Температура в 10:00',
        'unknown_23':'Температура в 10:00',
        'unknown_24':'4-8_GHz_stop',
        'unknown_25':'4-8_GHz_note',
        'unknown_26':'Callisto_start',
        'unknown_27':'Callisto_stop',
        'unknown_28':'Callisto_note',
        'unknown_29':'0,05-3_GHz_start',
        'unknown_30':'0,05-3_GHz_stop',
        'unknown_31':'0,05-3_GHz_note',
        
        'unknown_32':'3-24_GHz_start',
        'unknown_33':'3-24_GHz_stop',
        'unknown_34':'3-24_GHz_note',
        
        'unknown_35':'3-6_GHz_start',
        'unknown_36':'3-6_GHz_stop',
        'unknown_37':'3-6_GHz_note',
        
        'unknown_38':'6-12_GHz_start',
        'unknown_39':'6-12_GHz_stop',
        'unknown_40':'6-12_GHz_note',
        
        'unknown_41':'12-24_GHz_start',
        'unknown_42':'12-24_GHz_stop',
        'unknown_43':'12-24_GHz_note',

        'unknown_45':'2-24_GHz_start',
        'unknown_46':'2-24_GHz_stop',
        'unknown_47':'2-24_GHz_note',
        'general_note1':'4-8_GHz_note',
    }
    
    wide_order = ['Дата_время', 'Дата', 'Температура в 00:00', 'Погода в 00:00',  'Температура в 01:00', 'Погода в 01:00', 'Температура в 02:00', 'Погода в 02:00', 'Температура в 03:00', 'Погода в 03:00', 'Температура в 04:00', 'Погода в 04:00', 'Температура в 05:00', 'Погода в 05:00','Температура в 06:00', 'Погода в 06:00', 'Температура в 07:00', 'Погода в 07:00', 'Температура в 08:00', 'Погода в 08:00', 'Температура в 09:00', 'Погода в 09:00', 'Температура в 10:00', 'Погода в 10:00','4-8_GHz_start', '4-8_GHz_stop','4-8_GHz_note','Callisto_start','Callisto_stop','Callisto_note','0,05-3_GHz_start','0,05-3_GHz_stop','0,05-3_GHz_note','3-24_GHz_start','3-24_GHz_stop','3-24_GHz_note','3-6_GHz_start','3-6_GHz_stop','3-6_GHz_note','6-12_GHz_start','6-12_GHz_stop','6-12_GHz_note','12-24_GHz_start','12-24_GHz_stop','12-24_GHz_note','2-24_GHz_start','2-24_GHz_stop','2-24_GHz_note','Дежурный']
    
    df_wide_reorganized = reorganize_csv(
        'parsed_wide.csv',
        'parsed_wide_reorganized.csv',
        column_mapping=wide_mapping,
        column_order=wide_order
    )
    
    # Для длинного формата
    long_mapping = {
        'datetime': 'Дата_время',
        'date': 'Дата',
        'hour': 'Час',
        'temperature': 'Температура_°C',
        'weather': 'Погода',
        'general_note': 'Примечание',
        'duty_officer': 'Дежурный'
    }
    
    long_order = ['Дата_время', 'Дата', 'Час', 'Температура_°C', 'Погода', 'Примечание', 'Дежурный']
    
    df_long_reorganized = reorganize_csv(
        'parsed_long.csv',
        'parsed_long_reorganized.csv',
        column_mapping=long_mapping,
        column_order=long_order
    )