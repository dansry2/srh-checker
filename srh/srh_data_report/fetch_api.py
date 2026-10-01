import requests
from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import Optional
import json
import os

GRID_TO_GRATING = {
    5: "SRH0306",
    6: "SRH0612",
    7: "SRH1224",
}


def _get_api_url(api_url: Optional[str] = None) -> str:
    if api_url:
        return api_url.rstrip("/")
    env_url = os.environ.get("SRH_API_URL")
    if env_url:
        return env_url.rstrip("/")
    return "http://localhost:8000"


def _event_to_str(ev: dict) -> str:
    t = ev.get("type", "?")
    d = ev.get("date") or ""
    tm = ev.get("time") or ""
    note = ev.get("note") or ""

    if t == "breakdown":
        return f"Сломана с {d} {tm}".strip()
    elif t == "restore":
        return f"Починена {d} {tm}".strip()
    elif t == "other":
        de = ev.get("date_end") or ""
        te = ev.get("time_end") or ""
        if de or te:
            return f"Событие {d} {tm} – {de} {te}".strip()
        return f"Событие с {d} {tm}".strip()
    elif t == "deleted":
        return f"Удалено {d} {tm}".strip()
    return f"{t} {d} {tm}".strip()


def _build_display_status(events: list) -> str:
    if not events:
        return "OK"

    if any(ev.get("type") == "deleted" for ev in events):
        return "Удалено"

    breakdowns = [ev for ev in events if ev.get("type") == "breakdown"]
    restores = [ev for ev in events if ev.get("type") == "restore"]
    others = [ev for ev in events if ev.get("type") == "other"]
    open_others = [ev for ev in others if not ev.get("date_end")]

    parts = []

    if breakdowns:
        b = breakdowns[0]
        b_date = b.get("date") or ""
        b_time = b.get("time") or ""
        if restores:
            r = restores[0]
            r_date = r.get("date") or ""
            r_time = r.get("time") or ""
            parts.append(f"Сломана с {b_date} {b_time} до {r_date} {r_time}".strip())
        else:
            parts.append(f"Сломана с {b_date} {b_time} (ещё сломана)".strip())

    # Дедупликация
    seen = set()
    for o in open_others:
        key = (o.get("type"), o.get("date"), o.get("time"))
        if key in seen:
            continue
        seen.add(key)
        parts.append(_event_to_str(o))

    if not parts:
        return "OK"

    return "; ".join(parts)


def _build_antenna_entry(events: list, error: str = "", is_ok: bool = True) -> dict:
    if any(ev.get("type") == "deleted" for ev in events):
        return None

    has_breakdown = any(ev.get("type") == "breakdown" for ev in events)
    has_restore = any(ev.get("type") == "restore" for ev in events)
    has_open_other = any(
        ev.get("type") == "other" and not ev.get("date_end")
        for ev in events
    )

    if has_breakdown and not has_restore:
        status = "BROKEN"
    elif has_open_other:
        status = "EVENT"
    else:
        status = "OK"

    return {
        "status": status,
        "is_ok": is_ok,
        "error": error,
        "events": events,
        "display_status": _build_display_status(events),
    }


def fetch_antenna_journal(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    api_url: Optional[str] = None
) -> dict:
    if start_date is None:
        start_date = date.today()
    if end_date is None:
        end_date = date.today()

    api_url = _get_api_url(api_url)
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

    # Собираем все события по антеннам для заполнения пропусков
    antenna_events = defaultdict(list)  # (grating, antenna) -> [(date, event)]

    for day_entry in data:
        entry_date = datetime.fromisoformat(day_entry["date"]).date()
        grid_id = day_entry.get("grid_id")
        is_ok_range = day_entry.get("is_ok", True)

        grating = GRID_TO_GRATING.get(grid_id)
        if grating is None:
            continue

        antennas = {}
        for antenna_entry in day_entry.get("entries", []):
            antenna = antenna_entry.get("antenna", "?")
            error = antenna_entry.get("error", "")
            is_ok = antenna_entry.get("is_ok", True)
            events = antenna_entry.get("events") or []

            a = _build_antenna_entry(events, error, is_ok)
            if a is None:
                continue

            antennas[antenna] = a

            # Сохраняем события для заполнения пропусков
            for ev in events:
                antenna_events[(grating, antenna)].append((entry_date, ev))

        journal_data[entry_date][grating] = {
            "is_ok_range": is_ok_range,
            "antennas": antennas,
            "details": "; ".join(
                f"[{code}] {a['display_status']}"
                + (f": {a['error']}" if a['error'] else "")
                for code, a in antennas.items()
            )
        }

    # Заполняем пропущенные дни между breakdown и restore
    for (grating, antenna), events in antenna_events.items():
        # Находим breakdown и restore
        breakdown_ev = None
        restore_ev = None
        open_other_evs = []

        for d, ev in events:
            t = ev.get("type")
            if t == "breakdown" and breakdown_ev is None:
                breakdown_ev = (d, ev)
            elif t == "restore" and restore_ev is None:
                restore_ev = (d, ev)
            elif t == "other" and not ev.get("date_end"):
                open_other_evs.append((d, ev))

        if not breakdown_ev:
            continue

        b_date = breakdown_ev[0]
        r_date = restore_ev[0] if restore_ev else end_date

        # Заполняем пропущенные дни
        current = b_date
        while current <= r_date:
            if current not in journal_data:
                journal_data[current] = {}
            if grating not in journal_data[current]:
                journal_data[current][grating] = {
                    "is_ok_range": True,
                    "antennas": {},
                    "details": ""
                }

            # Проверяем, есть ли уже антенна на этот день
            if antenna not in journal_data[current][grating]["antennas"]:
                # Собираем события на этот день
                day_events = []
                if current == b_date:
                    day_events.append(breakdown_ev[1])
                elif current < r_date:
                    # Промежуточный день — breakdown переносится
                    day_events.append(breakdown_ev[1])

                if current == r_date and restore_ev:
                    day_events.append(restore_ev[1])

                # Открытые other
                for od, oev in open_other_evs:
                    if current >= od and (not restore_ev or current <= r_date):
                        if oev not in day_events:
                            day_events.append(oev)

                if day_events:
                    a = _build_antenna_entry(day_events)
                    if a:
                        journal_data[current][grating]["antennas"][antenna] = a

            # Пересчитываем details
            if journal_data[current][grating]["antennas"]:
                journal_data[current][grating]["details"] = "; ".join(
                    f"[{code}] {a['display_status']}"
                    + (f": {a['error']}" if a['error'] else "")
                    for code, a in journal_data[current][grating]["antennas"].items()
                )

            current += timedelta(days=1)

        # После r_date — заполняем дни с открытыми other до end_date
        if open_other_evs:
            fill_start = r_date + timedelta(days=1)
            current = fill_start
            while current <= end_date:
                if current not in journal_data:
                    journal_data[current] = {}
                if grating not in journal_data[current]:
                    journal_data[current][grating] = {
                        "is_ok_range": True,
                        "antennas": {},
                        "details": ""
                    }

                if antenna not in journal_data[current][grating]["antennas"]:
                    day_events = [oev for _, oev in open_other_evs]
                    a = _build_antenna_entry(day_events)
                    if a:
                        journal_data[current][grating]["antennas"][antenna] = a

                if journal_data[current][grating]["antennas"]:
                    journal_data[current][grating]["details"] = "; ".join(
                        f"[{code}] {a['display_status']}"
                        + (f": {a['error']}" if a['error'] else "")
                        for code, a in journal_data[current][grating]["antennas"].items()
                    )

                current += timedelta(days=1)

    return dict(journal_data)


def update_files_with_api(
    data_dir: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    api_url: Optional[str] = None
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
                    day_data[grating]["range_broken"] = not jdata.get("is_ok_range", True)
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
