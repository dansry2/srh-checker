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
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckResult':
        return cls(
            date=datetime.date.fromisoformat(data["date"]),
            checker_name=data["checker_name"],
            status=DataStatus(data["status"]),  
            details=data.get("details", {}),
            comment=data.get("comment", "")
        )


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
        self.gratings = ['SRH0612', 'SRH1224', 'SRH0306']
    
    def check_day(self, date: datetime.date) -> CheckResult:
        t1 = datetime.datetime.combine(date, datetime.time(self.start_hour, 0, 0))
        t2 = datetime.datetime.combine(date, datetime.time(self.end_hour, 0, 0))
        
        try:
            frequencies = srhimages.get_frequencies(t1, t2)
            availability = {}
            for grating in self.gratings:
                availability[grating] = (grating in frequencies and len(frequencies[grating]) > 0)
            
            if all(availability.values()):
                status = DataStatus.GOOD
                comment = "Все решётки доступны"
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
                details={"availability": availability},
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
    
    def __init__(self, 
                 start_hour: int = 1, 
                 end_hour: int = 9,
                 n_segments: int = 8,
                 valley_depth_threshold: float = 15,
                 slope_threshold: float = 0.01,
                 trend_significance: float = 0.05):
        super().__init__("quality")
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.n_segments = n_segments
        self.valley_depth_threshold = valley_depth_threshold
        self.slope_threshold = slope_threshold
        self.trend_significance = trend_significance
    
    def _get_all_frequencies(self, corr) -> List[int]:
        if hasattr(corr, 'frequencies') and corr.frequencies is not None:
            return sorted(corr.frequencies.tolist())
        return []
    
    def _smooth_data(self, data, window_length=None):
        if len(data) < 10:
            return data
        
        if window_length is None:
            window_length = min(11, len(data) - 1 if len(data) % 2 == 0 else len(data))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < 3:
                window_length = 3
        
        try:
            return savgol_filter(data, window_length, 2)
        except:
            return data
    
    def _analyze_trend(self, times, flux_I):
        n = len(flux_I)
        
        if n < 5:
            return {"direction": "stable", "slope": 0, "has_valley": False, 
                   "valley_depth": 0, "valley_position": 0, "is_problem": False}
        
        start_time = times[0]
        time_seconds = np.array([(t - start_time).total_seconds() for t in times])
        flux_smooth = self._smooth_data(flux_I)
        
        slope, _, _, p, _ = stats.linregress(time_seconds, flux_smooth)
        
        if p < self.trend_significance and abs(slope) > self.slope_threshold:
            direction = "down" if slope < 0 else "up"
        else:
            direction = "stable"
        
        min_idx = np.argmin(flux_smooth)
        min_value = flux_smooth[min_idx]
        start_value = flux_smooth[0]
        
        has_valley = False
        valley_depth = 0
        valley_position = min_idx / n
        
        if start_value > 0:
            valley_depth = (start_value - min_value) / start_value * 100
        
        if valley_depth > self.valley_depth_threshold and 0.2 < valley_position < 0.8:
            has_valley = True
        
        is_problem = False
        problem_reasons = []
        
        if direction == "down":
            is_problem = True
            problem_reasons.append(f"нисходящий тренд (slope={slope:.3f})")
        
        if has_valley:
            is_problem = True
            problem_reasons.append(f"галочка глубиной {valley_depth:.1f}%")
        
        return {
            "direction": direction,
            "slope": slope,
            "has_valley": has_valley,
            "valley_depth": valley_depth,
            "valley_position": valley_position,
            "is_problem": is_problem,
            "problem_reasons": problem_reasons,
            "n_points": n,
            "flux_median": float(np.median(flux_I)),
            "flux_std": float(np.std(flux_I))
        }
    
    def _check_frequency(self, date: datetime.date, array: str, frequency: int) -> Dict[str, Any]:
        try:
            corr = srhcp.SRHCorrPlot(date, array, frequency, "corrplot_cache")
            
            if corr.data is None:
                return {"status": DataStatus.NO_DATA, "comment": "Нет данных в FITS файле"}
            
            time_indices = [
                i for i, t in enumerate(corr.times) 
                if self.start_hour <= t.hour < self.end_hour
            ]
            
            if len(time_indices) < 50:
                return {
                    "status": DataStatus.NO_DATA,
                    "comment": f"Маловато точек: {len(time_indices)}",
                    "n_points": len(time_indices)
                }
            
            flux_I = corr.flux_I[time_indices]
            times = [corr.times[i] for i in time_indices]
            flux_I = np.nan_to_num(flux_I, nan=0.0)
            
            if np.max(flux_I) > 1e6:
                return {
                    "status": DataStatus.BAD,
                    "comment": f"Аномальные выбросы: max={np.max(flux_I):.1e}"
                }
            
            analysis = self._analyze_trend(times, flux_I)
            
            if analysis["is_problem"]:
                return {
                    "status": DataStatus.PROBLEM,
                    "comment": f"Проблема: {', '.join(analysis['problem_reasons'])}",
                    **analysis
                }
            
            return {
                "status": DataStatus.GOOD,
                "comment": f"Норма, flux={analysis['flux_median']:.1f} SFU",
                **analysis
            }
            
        except Exception as e:
            return {"status": DataStatus.NO_DATA, "comment": f"Ошибка: {str(e)}"}
    
    def check_day(self, date: datetime.date) -> CheckResult:
        gratings = ['SRH0612', 'SRH1224', 'SRH0306']
        
        try:
            test_corr = srhcp.SRHCorrPlot(date, gratings[0], 6000, "corrplot_cache")
            if test_corr.data is not None and hasattr(test_corr, 'frequencies'):
                all_frequencies = sorted(test_corr.frequencies.tolist())
            else:
                all_frequencies = [3000, 4000, 5000, 6000, 7000, 8000]
        except:
            all_frequencies = [3000, 4000, 5000, 6000, 7000, 8000]
        
        results = {}
        overall_status = DataStatus.GOOD
        
        for grating in gratings:
            results[grating] = {}
            for freq in all_frequencies:
                freq_result = self._check_frequency(date, grating, freq)
                results[grating][freq] = freq_result
                
                if freq_result["status"] in [DataStatus.BAD, DataStatus.PROBLEM]:
                    if overall_status == DataStatus.GOOD:
                        overall_status = freq_result["status"]
        
        details = {
            "gratings": results,
            "frequencies": all_frequencies,
            "n_frequencies": len(all_frequencies)
        }
        
        problem_count = sum(
            1 for g in results 
            for f in results[g] 
            if results[g][f]["status"] in [DataStatus.PROBLEM, DataStatus.BAD]
        )
        
        comment = f"Проверено {len(all_frequencies)} частот, {problem_count} проблем"
        
        return CheckResult(
            date=date,
            checker_name=self.name,
            status=overall_status,
            details=details,
            comment=comment
        )


class DataQualityManager:
    
    def __init__(self):
        self.checkers: Dict[str, DataChecker] = {}
        self.results: Dict[datetime.date, Dict[str, CheckResult]] = {}
    
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
    
    def save_to_json(self, filename: str = "data_quality.json"):
        data = {}
        for date, date_results in self.results.items():
            data[date.isoformat()] = {
                checker_name: result.to_dict()  
                for checker_name, result in date_results.items()
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Результаты сохранены в {filename}")
    
    def load_from_json(self, filename: str = "data_quality.json"):
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results = {}
        for date_str, checkers_data in data.items():
            date = datetime.date.fromisoformat(date_str)
            self.results[date] = {}
            for checker_name, result_data in checkers_data.items():
                self.results[date][checker_name] = CheckResult.from_dict(result_data)
        
        print(f"✅ Загружено {len(self.results)} дней из {filename}")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for date, date_results in sorted(self.results.items()):
            row = {"Date": date}
            for checker_name, result in date_results.items():
                row[checker_name] = result.status.value
            rows.append(row)
        
        return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    manager = DataQualityManager()
    
    manager.add_checker(AvailabilityChecker(start_hour=0, end_hour=10))
    manager.add_checker(QualityChecker(
        start_hour=1, 
        end_hour=9,
        n_segments=8,
        valley_depth_threshold=20,
        slope_threshold=0.0055
    ))
    
    start = datetime.date(2024, 5, 7)
    end = datetime.date(2024, 5, 12)
    
    manager.check_period(start, end)
    
    manager.save_to_json("data_quality_new.json")
    
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("="*70)
    print(manager.get_summary_dataframe())