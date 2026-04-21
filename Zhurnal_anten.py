import pandas as pd
import json
from datetime import datetime
import re
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

    journal_data = parse_antenna_journal(excel_path)
    print(f"Найдено записей для {len(journal_data)} дней")
    
    if not os.path.exists(data_dir):
        print(f"\nПапка не найдена: {data_dir}")
        return 0
    
    grating_order = ['SRH0306', 'SRH0612', 'SRH1224']
    
    updated_count = 0
    matched_dates = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            day_data = json.load(f)
        
        date_str = day_data.get("date", filename.replace('.json', ''))
        date_obj = datetime.fromisoformat(date_str).date()
        
        if date_obj in journal_data:
            ordered_notes = {}
            for grating in grating_order:
                if grating in journal_data[date_obj]:
                    ordered_notes[grating] = journal_data[date_obj][grating]
            for grating, note in journal_data[date_obj].items():
                if grating not in ordered_notes:
                    ordered_notes[grating] = note
            
            day_data["journal_notes"] = ordered_notes
            updated_count += 1
            matched_dates.append(date_str)
            print(f"  📝 {date_str}: добавлены примечания для {list(ordered_notes.keys())}")
        else:
            day_data["journal_notes"] = {}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(day_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nОбновлено {updated_count} файлов с примечаниями")
    
    return updated_count


def analyze_journal_coverage(data_dir: str):
    print("\n" + "="*70)
    print("АНАЛИЗ ПОКРЫТИЯ ЖУРНАЛА")
    print("="*70)
    
    grating_order = ['SRH0306', 'SRH0612', 'SRH1224']
    
    for grating in grating_order:
        print(f"\n🔍 {grating}:")
        
        days_with_notes = 0
        days_with_problems_in_spectral = 0
        days_with_both = 0
        days_with_only_journal = 0
        days_with_only_spectral = 0
        total_files = 0
        
        for filename in os.listdir(data_dir):
            if not filename.endswith('.json'):
                continue
            
            total_files += 1
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                day_data = json.load(f)
            
            has_journal_note = grating in day_data.get("journal_notes", {})
            has_spectral_problem = False
            
            if "spectral_analysis" in day_data and grating in day_data["spectral_analysis"]:
                for freq, freq_data in day_data["spectral_analysis"][grating].items():
                    if isinstance(freq_data, dict):
                        state = freq_data.get("state") or freq_data.get("status")
                        if state in ["PROBLEM", "BAD"]:
                            has_spectral_problem = True
                            break
            
            if has_journal_note:
                days_with_notes += 1
            if has_spectral_problem:
                days_with_problems_in_spectral += 1
            if has_journal_note and has_spectral_problem:
                days_with_both += 1
            elif has_journal_note and not has_spectral_problem:
                days_with_only_journal += 1
            elif not has_journal_note and has_spectral_problem:
                days_with_only_spectral += 1
        
        print(f"  Всего файлов: {total_files}")
        print(f"  Дней с примечаниями в журнале: {days_with_notes}")
        print(f"  Дней с проблемами в spectral_analysis: {days_with_problems_in_spectral}")
        print(f"  Дней с обоими проблемами: {days_with_both}")
        print(f"  Дней только с журналом: {days_with_only_journal}")
        print(f"  Дней только с spectral_analysis: {days_with_only_spectral}")


def merge_files_to_single_json(data_dir: str, output_path: str):
    merged_data = {}
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            date_str = filename.replace('.json', '')
            
            with open(filepath, 'r', encoding='utf-8') as f:
                day_data = json.load(f)
            
            merged_data[date_str] = day_data
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f" Объединённый файл сохранён: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ИНТЕГРАЦИЯ ЖУРНАЛА ОШИБОК АНТЕНН")
    print("="*70)
    
    data_dir = "data_quality_files"       
    excel_path = "Radioheliograph.xlsx"   
    
    if not os.path.exists(data_dir):
        print(f"\nПапка не найдена: {data_dir}")
        print("Доступные папки:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        exit(1)
    
    if not os.path.exists(excel_path):
        print(f"\n Excel файл не найден: {excel_path}")
        print(" Доступные Excel файлы:")
        for f in os.listdir('.'):
            if f.endswith(('.xlsx', '.xls')):
                print(f"  - {f}")
        exit(1)
    

    updated = update_files_with_journal(data_dir, excel_path, create_backup=True)
    
    if updated >= 0:
 
        analyze_journal_coverage(data_dir)
        
        print("\n" + "="*70)
        print(" ГОТОВО!")
        print("="*70)
        print(f"\n Файлы обновлены в: {data_dir}")
        
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if sample_files:
            sample_file = sample_files[0]
            print(f"\n Пример файла: {sample_file}")
            
            with open(os.path.join(data_dir, sample_file), 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            print(f"  - date: {sample_data.get('date')}")
            print(f"  - availability: {list(sample_data.get('availability', {}).keys())}")
            print(f"  - spectral_analysis: {list(sample_data.get('spectral_analysis', {}).keys())}")
            
            journal_keys = list(sample_data.get('journal_notes', {}).keys())
            if journal_keys:
                print(f"  - journal_notes: {journal_keys}")
                for grating, note in sample_data['journal_notes'].items():
                    print(f"      {grating}: {note[:100]}...")