import requests
from datetime import date, datetime
from collections import defaultdict
from typing import Optional
import json
import os

GRID_TO_GRATING = {
    5: "SRH0306",
    6: "SRH0612",
    7: "SRH1224",
}


def fetch_antenna_journal(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    api_url: str = "http://localhost:8000"
) -> dict:
    if start_date is None:
        start_date = date.today()
    if end_date is None:
        end_date = date.today()

    url = f"{api_url}/api/v1/errors"
    params = {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat()
    }

    print(f"Запрос к API: {url}")
    print(f"Период: {start_date} — {end_date}")

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        print(f"Ошибка: не удалось подключиться к {api_url}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return {}

    print(f"Получено записей: {len(data)}")

    journal_data = defaultdict(lambda: defaultdict(dict))

    for day_entry in data:
        entry_date = datetime.fromisoformat(day_entry["date"]).date()
        grid_id = day_entry.get("grid_id")
        is_ok_range = day_entry.get("is_ok", True)

        grating = GRID_TO_GRATING.get(grid_id)
        if grating is None:
            continue

        antennas = []
        for antenna_entry in day_entry.get("entries", []):
            antenna = antenna_entry.get("antenna", "?")
            error = antenna_entry.get("error", "")
            is_ok = antenna_entry.get("is_ok", True)
            
            status = "OK" if is_ok else "BROKEN"
            antennas.append({
                "antenna": antenna,
                "status": status,
                "is_ok": is_ok,
                "error": error
            })

        journal_data[entry_date][grating] = {
            "is_ok_range": is_ok_range,
            "antennas": antennas,
            "details": "; ".join(
                f"[{'OK' if a['is_ok'] else 'X'}] {a['antenna']}{': ' + a['error'] if a['error'] else ''}"
                for a in antennas
            )
        }

    return dict(journal_data)


def update_files_with_api(
    data_dir: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    api_url: str = "http://localhost:8000"
) -> int:
    print("Получение данных через JSON API...")
    journal_data = fetch_antenna_journal(start_date, end_date, api_url)

    if not journal_data:
        print("Нет данных для обновления")
        return 0

    if not os.path.exists(data_dir):
        print(f"Папка не найдена: {data_dir}")
        return 0

    updated_count = 0

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            day_data = json.load(f)

        date_str = day_data.get("date", filename.replace('.json', ''))
        date_obj = datetime.fromisoformat(date_str).date()

        if date_obj in journal_data:
            for grating, jdata in journal_data[date_obj].items():
                if grating in day_data:
                    day_data[grating]["range_broken"] = jdata["is_broken_range"]
                    day_data[grating]["journal_notes"] = {
                        "details": jdata["details"],
                        "antennas": jdata["antennas"]
                    }
                    print(f"  {date_str} / {grating}: добавлено")

            updated_count += 1

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(day_data, f, ensure_ascii=False, indent=2)

    print(f"Обновлено {updated_count} файлов")
    return updated_count
