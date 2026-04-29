import pandas as pd
import json
from datetime import datetime
from collections import defaultdict
import os


def parse_antenna_journal(excel_path: str) -> dict:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Файл не найден: {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name='Журнал ошибок антенн')
    
    journal_data = defaultdict(lambda: defaultdict(str))
    
    current_date = None
    current_grating = None
    
    grating_mapping = {
        '6-12GHz': 'SRH0612',
        '6_12GHz': 'SRH0612',
        '06-12GHz': 'SRH0612',
        '12_24GHz': 'SRH1224',
        '12-24GHz': 'SRH1224',
        '3_6GHz': 'SRH0306',
        '3-6GHz': 'SRH0306',
        '3-6 GHz': 'SRH0306',
        'рем/Микран': None,
        'Микран': None,
        'по спирали': None,
    }
    
    for idx, row in df.iterrows():
        date_val = row.iloc[0] if len(row) > 0 else None
        grating_val = row.iloc[1] if len(row) > 1 else None
        note_val = row.iloc[2] if len(row) > 2 else None
        
        if pd.isna(date_val) or (isinstance(date_val, str) and str(date_val).startswith('#')):
            if isinstance(grating_val, str):
                grating_clean = grating_val.strip()
                if grating_clean in grating_mapping:
                    mapped = grating_mapping[grating_clean]
                    if mapped and current_date:
                        current_grating = mapped
                        
                        if isinstance(note_val, str) and note_val.strip() and note_val.strip() != '0':
                            existing = journal_data[current_date][current_grating]
                            if existing:
                                journal_data[current_date][current_grating] = existing + "; " + note_val.strip()
                            else:
                                journal_data[current_date][current_grating] = note_val.strip()
            continue
        
        if isinstance(date_val, (datetime, pd.Timestamp)):
            current_date = date_val.date()
        
        if isinstance(grating_val, str):
            grating_clean = grating_val.strip()
            if grating_clean in grating_mapping:
                mapped = grating_mapping[grating_clean]
                if mapped:
                    current_grating = mapped
        
        if isinstance(note_val, str) and note_val.strip() and note_val.strip() != '0':
            if current_date and current_grating:
                existing = journal_data[current_date][current_grating]
                if existing:
                    journal_data[current_date][current_grating] = existing + "; " + note_val.strip()
                else:
                    journal_data[current_date][current_grating] = note_val.strip()
    
    return dict(journal_data)


def update_files_with_journal(data_dir: str, excel_path: str, create_backup: bool = True):

    print(" Парсинг журнала ошибок...")
    journal_data = parse_antenna_journal(excel_path)
    print(f" Найдено записей для {len(journal_data)} дней")
    
    if not os.path.exists(data_dir):
        print(f"\n Папка не найдена: {data_dir}")
        return 0
    
    
    updated_count = 0
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            day_data = json.load(f)
        
        date_str = day_data.get("date", filename.replace('.json', ''))
        date_obj = datetime.fromisoformat(date_str).date()
        
        if date_obj in journal_data:
            for grating, notes_text in journal_data[date_obj].items():
                if grating in day_data:
                    day_data[grating]["journal_notes"] = {"details": notes_text}
                    print(f"  📝 {date_str} / {grating}: добавлены свёрнутые примечания")
            
            updated_count += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(day_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n Обновлено {updated_count} файлов")
    print(f" Файлы обновлены в '{data_dir}'")
    
    return updated_count


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ИНТЕГРАЦИЯ ЖУРНАЛА ОШИБОК АНТЕНН")
    print("="*70)
    
    data_dir = "data_quality_files"
    excel_path = "Radioheliograph.xlsx"
    
    if not os.path.exists(data_dir):
        print(f"\n Папка не найдена: {data_dir}")
        exit(1)
    
    if not os.path.exists(excel_path):
        print(f"\n Excel файл не найден: {excel_path}")
        exit(1)
    
    updated = update_files_with_journal(data_dir, excel_path, create_backup=True)
    
    if updated >= 0:
        print("\n" + "="*70)
        print(" ГОТОВО!")
        print("="*70)
        
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if sample_files:
            sample_file = sample_files[0]
            print(f"\n🔍 Пример файла: {sample_file}")
            
            with open(os.path.join(data_dir, sample_file), 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            print(f"\n  date: {sample_data.get('date')}")
            for grating_name in ['SRH0306', 'SRH0612', 'SRH1224']:
                if grating_name in sample_data:
                    grating_data = sample_data[grating_name]
                    print(f"\n  📡 {grating_name}:")
                    print(f"     - availability: {grating_data.get('availability')}")
                    print(f"     - flux: {len(grating_data.get('flux', {}))} частот")
                    journal = grating_data.get('journal_notes', '')
                    if journal:
                        print(f"     - journal_notes: {journal[:100]}...")
                    else:
                        print(f"     - journal_notes: нет")