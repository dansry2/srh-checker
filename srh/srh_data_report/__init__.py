"""Пакет проверки данных СРХ."""
import os
from datetime import date
from .report import CheckReport
from .checks import AvailabilityChecker, QualityChecker, DataQualityManager
from .Zhurnal_anten import parse_antenna_journal

__version__ = "0.1.2"


def check_day(check_date: date, start_hour: int = 0, end_hour: int = 10,
              excel_path: str = None) -> CheckReport:
    """
    Проверяет данные за один день.
    Если указан excel_path — добавляет journal_notes из журнала антенн.
    """
    manager = DataQualityManager(output_dir="/tmp/srh_check")
    manager.add_checker(AvailabilityChecker(start_hour=start_hour, end_hour=end_hour))
    manager.add_checker(QualityChecker(start_hour=1, end_hour=9))
    
    manager.check_period(check_date, check_date)
    
    raw_data = {}
    for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
        raw_data[grating] = {
            "availability": False,
            "time_range": "NO_DATA",
            "flux": {}
        }
    
    if check_date in manager.results:
        if "availability" in manager.results[check_date]:
            avail = manager.results[check_date]["availability"].to_dict()
            for grating, is_avail in avail["details"]["availability"].items():
                raw_data[grating]["availability"] = is_avail
            for grating, tc in avail["details"]["time_checks"].items():
                raw_data[grating]["time_range"] = tc.get("time_range", "NO_DATA")
        
        if "flux" in manager.results[check_date]:
            flux = manager.results[check_date]["flux"].to_dict()
            for grating, freqs in flux["details"]["flux"].items():
                raw_data[grating]["flux"] = freqs
    
    if excel_path and os.path.exists(excel_path):
        journal_data = parse_antenna_journal(excel_path)
        if check_date in journal_data:
            for grating, notes in journal_data[check_date].items():
                if grating in raw_data:
                    raw_data[grating]["journal_notes"] = {"details": notes}
    
    return CheckReport(check_date, raw_data)