import srhimages
import datetime
import pandas as pd
import os
import numpy as np
import json
import srhcp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
from scipy import stats
from scipy.signal import savgol_filter
import astropy.constants as const
import astropy.units as u
from srhimages.helpers.zirin_tb import SRHQSunTb


class DataStatus(Enum):
    GOOD = "GOOD"           
    PROBLEM = "PROBLEM"     
    BAD = "BAD"             
    NO_DATA = "NO_DATA"   
    NOT_CHECKED = "NOT_CHECKED"


@dataclass
class CheckResult:
    date: datetime.date
    checker_name: str
    status: DataStatus
    details: Dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    
    def to_dict(self) -> Dict:
        def convert_to_serializable(obj):
            if isinstance(obj, DataStatus):
                return obj.value
            elif isinstance(obj, datetime.date):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        return {
            "date": self.date.isoformat(),
            "checker_name": self.checker_name,
            "status": self.status.value, 
            "details": convert_to_serializable(self.details),
            "comment": self.comment
        }


class DataChecker(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check_day(self, date: datetime.date) -> CheckResult:
        pass
    
    def check_period(self, start_date: datetime.date, end_date: datetime.date) -> List[CheckResult]:
        results = []
        current = start_date
        while current <= end_date:
            result = self.check_day(current)
            results.append(result)
            current += datetime.timedelta(days=1)
        return results


class AvailabilityChecker(DataChecker):
    
    def __init__(self, start_hour: int = 0, end_hour: int = 10):
        super().__init__("availability")
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.gratings = ['SRH0306', 'SRH0612', 'SRH1224']
    
    def _check_time_sequence(self, times: List[datetime.datetime]) -> Dict[str, Any]:

        if len(times) < 2:
            return {
                "time_range": "NO_DATA",
                "time_issues": [],
                "time_start": times[0].strftime('%H:%M:%S') if times else None,
                "time_end": times[-1].strftime('%H:%M:%S') if times else None
            }
        
        issues = []
        
        for i in range(1, len(times)):
            time_diff = (times[i] - times[i-1]).total_seconds()
            
            if time_diff < 0:
                issues.append(f"Скачок назад на {abs(time_diff):.1f} сек в {times[i-1].strftime('%H:%M:%S')} -> {times[i].strftime('%H:%M:%S')}")
            
            elif time_diff > 3600:
                issues.append(f"Большой промежуток {time_diff/3600:.1f} часов между {times[i-1].strftime('%H:%M:%S')} и {times[i].strftime('%H:%M:%S')}")
        
        if len(issues) == 0:
            time_range_status = "GOOD"
        elif any("Скачок назад" in issue for issue in issues):
            time_range_status = "BAD"
        else:
            time_range_status = "PROBLEM"
        
        return {
            "time_range": time_range_status,
            "time_issues": issues,
            "time_start": times[0].strftime('%H:%M:%S'),
            "time_end": times[-1].strftime('%H:%M:%S'),
            "total_points": len(times),
            "time_span_hours": (times[-1] - times[0]).total_seconds() / 3600
        }
    
    def check_day(self, date: datetime.date) -> CheckResult:
        t1 = datetime.datetime.combine(date, datetime.time(self.start_hour, 0, 0))
        t2 = datetime.datetime.combine(date, datetime.time(self.end_hour, 0, 0))
        
        try:
            frequencies = srhimages.get_frequencies(t1, t2)
            availability = {}
            time_checks = {}
            
            for grating in self.gratings:
                availability[grating] = (grating in frequencies and len(frequencies[grating]) > 0)
                
                if availability[grating]:
                    try:
                      
                        test_corr = srhcp.SRHCorrPlot(date, grating, None, "corrplot_cache")
                        if test_corr.data is not None and hasattr(test_corr, 'times'):
                            time_check = self._check_time_sequence(test_corr.times)
                            time_checks[grating] = time_check
                        else:
                            time_checks[grating] = {
                                "time_range": "NO_DATA",
                                "time_issues": ["Не удалось получить временные метки"]
                            }
                    except Exception as e:
                        time_checks[grating] = {
                            "time_range": "NO_DATA",
                            "time_issues": [f"Ошибка проверки времени: {str(e)}"]
                        }
                else:
                    time_checks[grating] = {
                        "time_range": "NO_DATA",
                        "time_issues": ["Решетка недоступна"]
                    }
            
            if all(availability.values()):
                status = DataStatus.GOOD
                comment = "Все решётки доступны"
                
                bad_time_count = sum(1 for tc in time_checks.values() if tc.get("time_range") == "BAD")
                if bad_time_count > 0:
                    status = DataStatus.BAD
                    comment += f", но проблемы с временной последовательностью в {bad_time_count} решётках"
                    
            elif any(availability.values()):
                status = DataStatus.PROBLEM
                missing = [g for g, avail in availability.items() if not avail]
                comment = f"Отсутствуют решётки: {', '.join(missing)}"
            else:
                status = DataStatus.NO_DATA
                comment = "Нет данных ни для одной решётки"
            
            return CheckResult(
                date=date,
                checker_name=self.name,
                status=status,
                details={
                    "availability": availability,
                    "time_checks": time_checks
                },
                comment=comment
            )
            
        except Exception as e:
            return CheckResult(
                date=date,
                checker_name=self.name,
                status=DataStatus.NO_DATA,
                details={"error": str(e)},
                comment=f"Ошибка: {e}"
            )


class QualityChecker(DataChecker):
    
    def __init__(
        self, 
        start_hour: int = 1, 
        end_hour: int = 9,
        sfu_min_ratio: float = 0.9,
        sfu_max_ratio: float = 10.0,
        anomaly_threshold_percent: float = 50, 
        anomaly_duration_minutes: int = 40  
    ):
        super().__init__("flux")
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.sfu_min_ratio = sfu_min_ratio
        self.sfu_max_ratio = sfu_max_ratio
        self.anomaly_threshold_percent = anomaly_threshold_percent
        self.anomaly_duration_minutes = anomaly_duration_minutes
    
    def _calculate_sfu(self, freq_mhz: float) -> float:

        try:
            freq_ghz = freq_mhz / 1000.0
            f = freq_ghz * 1e9 * u.hertz
            
            Tb = SRHQSunTb.get_tb(freq_ghz) * u.Kelvin
            
            sun_angle = 6.794 * 1e-5
            
            energy = 2 * const.k_B * Tb * (f / const.c)**2 * sun_angle
            
            sfu = 1e-22 * u.watt * u.meter**-2 * u.hertz**-1
            
            return float((energy / sfu).si)
        except Exception as e:
            print(f"Ошибка при расчете SFU для {freq_mhz} МГц: {e}")
            return None
    
    def _check_sfu_thresholds(self, flux_median: float, freq_mhz: int) -> Dict[str, Any]:
        expected_sfu = self._calculate_sfu(freq_mhz)
        
        if expected_sfu is None:
            return {
                "sfu_check_passed": True,
                "sfu_reason": "Не удалось рассчитать SFU",
                "expected_sfu": None
            }
        
        min_allowed = expected_sfu * self.sfu_min_ratio
        max_allowed = expected_sfu * self.sfu_max_ratio
        
        checks_passed = True
        reasons = []
        
        if flux_median < min_allowed:
            checks_passed = False
            reasons.append(f"Медиана ({flux_median:.1f} SFU) меньше {self.sfu_min_ratio*100:.0f}% от SFU ({expected_sfu:.1f} SFU)")
        
        if flux_median > max_allowed:
            checks_passed = False
            reasons.append(f"Медиана ({flux_median:.1f} SFU) превышает SFU ({expected_sfu:.1f} SFU) в {flux_median/expected_sfu:.1f} раз")
        
        return {
            "sfu_check_passed": checks_passed,
            "expected_sfu": expected_sfu,
            "min_allowed": min_allowed,
            "max_allowed": max_allowed,
            "flux_median_sfu": flux_median,
            "ratio_to_sfu": flux_median / expected_sfu if expected_sfu > 0 else None,
            "sfu_reason": ", ".join(reasons) if reasons else "Поток в норме относительно SFU"
        }
    
    def _find_anomalies(self, times: List[datetime.datetime], flux_I: np.ndarray, median_value: float) -> Dict[str, Any]:

        if len(flux_I) < 2:
            return {
                "has_anomalies": False,
                "anomaly_periods": []
            }
        
        lower_threshold = median_value * (1 - self.anomaly_threshold_percent / 100)
        
        anomaly_periods = []
        in_anomaly = False
        anomaly_start_idx = 0
        
        for i in range(len(flux_I)):
            is_anomaly = flux_I[i] < lower_threshold
            
            if is_anomaly and not in_anomaly:
                
                in_anomaly = True
                anomaly_start_idx = i
            elif not is_anomaly and in_anomaly:
               
                in_anomaly = False
                
                duration_seconds = (times[i-1] - times[anomaly_start_idx]).total_seconds()
                duration_minutes = duration_seconds / 60
                
                if duration_minutes >= self.anomaly_duration_minutes:
                    anomaly_data = flux_I[anomaly_start_idx:i]
                    anomaly_periods.append({
                        "start_time": times[anomaly_start_idx].strftime('%H:%M:%S'),
                        "end_time": times[i-1].strftime('%H:%M:%S'),
                        "duration_minutes": round(duration_minutes, 1),
                        "min_value": float(np.min(anomaly_data)),
                        "max_value": float(np.max(anomaly_data)),
                        "mean_value": float(np.mean(anomaly_data)),
                        "median_value": float(np.median(anomaly_data)),
                        "deviation_percent": round(abs(np.mean(anomaly_data) - median_value) / median_value * 100, 1)
                    })
        
        if in_anomaly:
            duration_seconds = (times[-1] - times[anomaly_start_idx]).total_seconds()
            duration_minutes = duration_seconds / 60
            
            if duration_minutes >= self.anomaly_duration_minutes:
                anomaly_data = flux_I[anomaly_start_idx:]
                anomaly_periods.append({
                    "start_time": times[anomaly_start_idx].strftime('%H:%M:%S'),
                    "end_time": times[-1].strftime('%H:%M:%S'),
                    "duration_minutes": round(duration_minutes, 1),
                    "min_value": float(np.min(anomaly_data)),
                    "max_value": float(np.max(anomaly_data)),
                    "mean_value": float(np.mean(anomaly_data)),
                    "median_value": float(np.median(anomaly_data)),
                    "deviation_percent": round(abs(np.mean(anomaly_data) - median_value) / median_value * 100, 1)
                })
        
        return {
            "has_anomalies": len(anomaly_periods) > 0,
            "anomaly_periods": anomaly_periods,
            "threshold_low": lower_threshold
        }
    
    def _check_frequency(self, date: datetime.date, array: str, frequency: int) -> Dict[str, Any]:
        try:
            corr = srhcp.SRHCorrPlot(date, array, frequency, "corrplot_cache")
            
            if corr.data is None:
                return {"state": DataStatus.NO_DATA.value, "comment": "Нет данных в FITS файле"}
            
            time_indices = [
                i for i, t in enumerate(corr.times) 
                if self.start_hour <= t.hour < self.end_hour
            ]
            
            if len(time_indices) < 50:
                return {
                    "state": DataStatus.NO_DATA.value,
                    "comment": f"Маловато точек: {len(time_indices)}",
                    "n_points": len(time_indices)
                }
            
            flux_I = corr.flux_I[time_indices]
            times = [corr.times[i] for i in time_indices]
            flux_I = np.nan_to_num(flux_I, nan=0.0)
            
            if np.max(flux_I) > 1e6:
                return {
                    "state": DataStatus.BAD.value,
                    "comment": f"Аномальные выбросы: max={np.max(flux_I):.1e}"
                }
            
            flux_median = float(np.median(flux_I))
            flux_std = float(np.std(flux_I))
            flux_mean = float(np.mean(flux_I))
            n_points = len(flux_I)
            
            sfu_check = self._check_sfu_thresholds(flux_median, frequency)
            
            anomaly_check = self._find_anomalies(times, flux_I, flux_median)
            
            all_reasons = []
            
            if not sfu_check["sfu_check_passed"]:
                all_reasons.append(sfu_check["sfu_reason"])
            
            if anomaly_check["has_anomalies"]:
                for period in anomaly_check["anomaly_periods"]:
                    all_reasons.append(
                        f"Провал данных на {period['deviation_percent']}% от медианы "
                        f"({period['duration_minutes']} мин: "
                        f"{period['start_time']}-{period['end_time']})"
                    )
            
            if not sfu_check["sfu_check_passed"]:
                status = DataStatus.BAD.value  
            elif anomaly_check["has_anomalies"]:
                status = DataStatus.PROBLEM.value  
            else:
                status = DataStatus.GOOD.value
            
            result = {
                "state": status,
                "comment": "; ".join(all_reasons) if all_reasons else "OK",
                "time_start": f"{times[0].strftime('%H:%M')}",
                "time_range": f"{times[-1].strftime('%H:%M')}",
                "flux_I_median": flux_median,
                "flux_I_mean": flux_mean,
                "expected_sfu": sfu_check.get("expected_sfu"),
                "sfu_ratio": sfu_check.get("ratio_to_sfu"),
            }
            
            return result
            
        except Exception as e:
            return {"state": DataStatus.NO_DATA.value, "comment": f"Ошибка: {str(e)}"}
    
    def check_day(self, date: datetime.date) -> CheckResult:
        gratings = ['SRH0306', 'SRH0612', 'SRH1224']
        
        grating_frequencies = {}
        
        for grating in gratings:
            try:
                test_corr = srhcp.SRHCorrPlot(date, grating, None, "corrplot_cache")
                if test_corr.data is not None and hasattr(test_corr, 'frequencies'):
                    freqs = sorted(test_corr.frequencies.tolist())
                    grating_frequencies[grating] = freqs
            except:
                if grating == 'SRH0306':
                    freqs = [2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800]
                elif grating == 'SRH0612':
                    freqs = [6000, 6400, 6800, 7200, 7600, 8000, 8400, 8800, 9200, 9600, 10000, 10400, 10800, 11200, 11600, 12000]
                elif grating == 'SRH1224':
                    freqs = [12200, 12960, 13720, 14480, 15240, 16000, 16760, 17520, 18280, 19040, 19800, 20560, 21320, 22080, 23000, 23400]
                grating_frequencies[grating] = freqs
        
        results = {}
        overall_status = DataStatus.GOOD
        
        for grating in gratings:
            results[grating] = {}
            freqs = grating_frequencies.get(grating, [])
            for freq in freqs:
                freq_result = self._check_frequency(date, grating, freq)
                results[grating][str(freq)] = freq_result
                
                if freq_result.get("state") == DataStatus.BAD.value:
                    overall_status = DataStatus.BAD
                elif freq_result.get("state") == DataStatus.PROBLEM.value and overall_status != DataStatus.BAD:
                    overall_status = DataStatus.PROBLEM
        
        details = {
            "flux": results
        }
        
        problem_count = sum(
            1 for g in results 
            for f in results[g] 
            if results[g][f].get("state") in [DataStatus.PROBLEM.value, DataStatus.BAD.value]
        )
        
        total_freqs = sum(len(freqs) for freqs in grating_frequencies.values())
        comment = f"Проверено {total_freqs} частот, {problem_count} проблем"
        
        return CheckResult(
            date=date,
            checker_name=self.name,
            status=overall_status,
            details=details,
            comment=comment
        )


class DataQualityManager:
    
    def __init__(self, output_dir: str = "data_quality"):
        self.checkers: Dict[str, DataChecker] = {}
        self.results: Dict[datetime.date, Dict[str, CheckResult]] = {}
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f" Создана папка: {output_dir}")
    
    def add_checker(self, checker: DataChecker):
        self.checkers[checker.name] = checker
    
    def check_period(self, start_date: datetime.date, end_date: datetime.date):
        for checker_name, checker in self.checkers.items():
            print(f"\n{'='*70}")
            print(f"ЗАПУСК ПРОВЕРКИ: {checker_name}")
            print('='*70)
            
            results = checker.check_period(start_date, end_date)
            
            for result in results:
                if result.date not in self.results:
                    self.results[result.date] = {}
                self.results[result.date][checker_name] = result
                
                status_icon = {
                    DataStatus.GOOD: "✅",
                    DataStatus.PROBLEM: "⚠️",
                    DataStatus.BAD: "❌",
                    DataStatus.NO_DATA: "❓"
                }.get(result.status, "❓")
                
                print(f"{result.date}: {status_icon} {result.status.value} - {result.comment}")
                
                if checker_name == "availability":
                    details = result.to_dict()["details"]
                    time_checks = details.get("time_checks", {})
                    for grating, tc in time_checks.items():
                        time_status = tc.get("time_range", "UNKNOWN")
                        time_issues = tc.get("time_issues", [])
                        if time_issues:
                            print(f"  {grating} time_range: {time_status} - {', '.join(time_issues)}")
    
    def save_to_files(self):
        saved_count = 0
        
        for date, date_results in self.results.items():
            day_dict = {"date": date.isoformat()}
            
            for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
                day_dict[grating] = {
                    "availability": False,
                    "time_range": "NO_DATA",
                    "flux": {}
                }
            
            if "availability" in date_results:
                avail_details = date_results["availability"].to_dict()["details"]
                availability = avail_details.get("availability", {})
                time_checks = avail_details.get("time_checks", {})
                
                for grating, is_available in availability.items():
                    if grating in day_dict:
                        day_dict[grating]["availability"] = is_available
                        if grating in time_checks:
                            day_dict[grating]["time_range"] = time_checks[grating].get("time_range", "NO_DATA")
            
            if "flux" in date_results:
                spectral_details = date_results["flux"].to_dict()["details"]
                spectral_data = spectral_details.get("flux", {})
                for grating, freq_data in spectral_data.items():
                    if grating in day_dict:
                        day_dict[grating]["flux"] = freq_data
            
            filename = os.path.join(self.output_dir, f"{date.isoformat()}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(day_dict, f, ensure_ascii=False, indent=2)
            
            saved_count += 1
        
        print(f"\n Сохранено {saved_count} файлов в папку '{self.output_dir}'")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for date, date_results in sorted(self.results.items()):
            row = {"Date": date}
            for checker_name, result in date_results.items():
                row[checker_name] = result.status.value
                
                if checker_name == "availability":
                    details = result.to_dict()["details"]
                    time_checks = details.get("time_checks", {})
                    for grating, tc in time_checks.items():
                        row[f"{grating}_time_range"] = tc.get("time_range", "NO_DATA")
            
            rows.append(row)
        
        return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    manager = DataQualityManager(output_dir="data_quality_files")
    
    manager.add_checker(AvailabilityChecker(start_hour=0, end_hour=10))
    manager.add_checker(QualityChecker(
        start_hour=1, 
        end_hour=9,
        sfu_min_ratio=0.9,
        sfu_max_ratio=10.0,
        anomaly_threshold_percent=40,  
        anomaly_duration_minutes=40    
    ))
    
    start = datetime.date(2024, 5, 1)
    end = datetime.date(2024, 5, 30)
    
    manager.check_period(start, end)
    
    manager.save_to_files()
    
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("="*70)
    print(manager.get_summary_dataframe())