import srhimages
import datetime
import pandas as pd
import os
import numpy as np
import json
import srhcp
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from scipy import stats
from scipy.signal import savgol_filter  


class DataStatus(Enum):
    """Статусы качества данных"""
    GOOD = "GOOD"           
    PROBLEM = "PROBLEM"     
    BAD = "BAD"             
    NO_DATA = "NO_DATA"   
    NOT_CHECKED = "NOT_CHECKED"  


@dataclass
class GratingCheckResult:

    frequency: int
    state: DataStatus
    comment: str = ""
    flux_I_median: Optional[float] = None
    flux_I_std: Optional[float] = None
    trend_slope: Optional[float] = None      # общий наклон тренда
    trend_direction: Optional[str] = None     # "up", "down", "stable"
    has_valley: Optional[bool] = None         # есть ли "галочка"
    valley_depth_pct: Optional[float] = None  # глубина галочки в процентах
    valley_position: Optional[float] = None   # позиция галочки (0-1)
    local_trends: Optional[List[Dict]] = None # локальные тренды по сегментам
    n_points: int = 0
    time_range: str = ""
    
    def to_dict(self) -> Dict:

        result = {
            "state": self.state.value,
            "comment": self.comment,
            "frequency": self.frequency,
            "n_points": self.n_points,
            "time_range": self.time_range
        }
        
        if self.flux_I_median is not None:
            result["flux_I_median"] = round(self.flux_I_median, 1)
        if self.flux_I_std is not None:
            result["flux_I_std"] = round(self.flux_I_std, 1)
        if self.trend_slope is not None:
            result["trend_slope"] = round(self.trend_slope, 3)
        if self.trend_direction is not None:
            result["trend_direction"] = self.trend_direction
        if self.has_valley is not None:
            result["has_valley"] = self.has_valley
        if self.valley_depth_pct is not None:
            result["valley_depth_pct"] = round(self.valley_depth_pct, 1)
        if self.valley_position is not None:
            result["valley_position"] = round(self.valley_position, 2)
        if self.local_trends is not None:
            result["local_trends"] = self.local_trends
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GratingCheckResult':

        return cls(
            frequency=data.get("frequency", 0),
            state=DataStatus(data.get("state", "NOT_CHECKED")),
            comment=data.get("comment", ""),
            flux_I_median=data.get("flux_I_median"),
            flux_I_std=data.get("flux_I_std"),
            trend_slope=data.get("trend_slope"),
            trend_direction=data.get("trend_direction"),
            has_valley=data.get("has_valley"),
            valley_depth_pct=data.get("valley_depth_pct"),
            valley_position=data.get("valley_position"),
            local_trends=data.get("local_trends"),
            n_points=data.get("n_points", 0),
            time_range=data.get("time_range", "")
        )


@dataclass
class DayCheckResult:
    date: datetime.date
    availability: Dict[str, bool]
    quality: Dict[str, Dict[int, GratingCheckResult]]
    all_frequencies: Dict[str, List[int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            "date": self.date.isoformat(),
            "availability": self.availability,
            "quality": {},
            "all_frequencies": self.all_frequencies
        }
        
        for grating, freq_dict in self.quality.items():
            result["quality"][grating] = {
                str(freq): check.to_dict() 
                for freq, check in freq_dict.items()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DayCheckResult':
        quality = {}
        for grating, freq_dict in data.get("quality", {}).items():
            quality[grating] = {
                int(freq): GratingCheckResult.from_dict(freq_data)
                for freq, freq_data in freq_dict.items()
            }
        
        return cls(
            date=datetime.date.fromisoformat(data["date"]),
            availability=data.get("availability", {}),
            quality=quality,
            all_frequencies=data.get("all_frequencies", {})
        )
    
    def get_summary(self, grating: str, agg_method: str = "worst") -> DataStatus:
        if not self.availability.get(grating, False):
            return DataStatus.NO_DATA
        
        freq_results = self.quality.get(grating, {})
        
        if not freq_results:
            return DataStatus.NOT_CHECKED
        
        if agg_method == "worst":
            status_priority = {
                DataStatus.BAD: 3,
                DataStatus.PROBLEM: 2,
                DataStatus.GOOD: 1,
                DataStatus.NO_DATA: 0,
                DataStatus.NOT_CHECKED: 0
            }
            
            worst_status = DataStatus.GOOD
            for result in freq_results.values():
                if status_priority[result.state] > status_priority[worst_status]:
                    worst_status = result.state
            
            return worst_status
        
        return DataStatus.NOT_CHECKED
    
    def get_bad_frequencies(self, grating: str) -> List[int]:
        bad_freqs = []
        for freq, result in self.quality.get(grating, {}).items():
            if result.state in [DataStatus.PROBLEM, DataStatus.BAD]:
                bad_freqs.append(freq)
        return bad_freqs
    
    def has_problem(self) -> bool:
        for grating in self.availability:
            if self.get_bad_frequencies(grating):
                return True
        return False



class QualityChecker:
    
    def __init__(self, 
                 start_hour: int = 1, 
                 end_hour: int = 9,
                 outlier_threshold: float = 1e6,
                 n_segments: int = 8,               
                 valley_depth_threshold: float = 15, 
                 slope_threshold: float = 0.01,      
                 trend_significance: float = 0.05):  

        self.start_hour = start_hour
        self.end_hour = end_hour
        self.outlier_threshold = outlier_threshold
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
    
    def _detect_local_trends(self, times, flux_I):
 
        n = len(flux_I)
        
        if n < self.n_segments * 3:  
        
            return self._simple_trend_analysis(times, flux_I)
        
        flux_smooth = self._smooth_data(flux_I)
        
        start_time = times[0]
        time_seconds = np.array([(t - start_time).total_seconds() for t in times])
        
        slope_global, _, _, p_global, _ = stats.linregress(time_seconds, flux_smooth)
        
        if p_global < self.trend_significance and abs(slope_global) > self.slope_threshold:
            if slope_global < 0:
                global_direction = "down"
            else:
                global_direction = "up"
        else:
            global_direction = "stable"
        
        segment_size = n // self.n_segments
        local_trends = []
        
        for i in range(self.n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, n)
            
            if end_idx - start_idx < 5:
                continue
            
            seg_times = time_seconds[start_idx:end_idx]
            seg_flux = flux_smooth[start_idx:end_idx]
            
            slope_seg, _, _, p_seg, _ = stats.linregress(seg_times, seg_flux)
            
            if p_seg < self.trend_significance and abs(slope_seg) > self.slope_threshold:
                if slope_seg < 0:
                    seg_direction = "down"
                else:
                    seg_direction = "up"
            else:
                seg_direction = "stable"
            
            local_trends.append({
                "segment": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "slope": slope_seg,
                "direction": seg_direction,
                "significant": p_seg < self.trend_significance
            })
        
        has_valley = False
        valley_depth = 0
        valley_position = 0
        
        min_idx = np.argmin(flux_smooth)
        min_value = flux_smooth[min_idx]
        start_value = flux_smooth[0]
        end_value = flux_smooth[-1]
        
        if start_value > 0:
            valley_depth = (start_value - min_value) / start_value * 100
        
        valley_position = min_idx / n
        
        if (valley_depth > self.valley_depth_threshold and 
            0.2 < valley_position < 0.8):
            
            has_down_before = False
            has_up_after = False
            
            for trend in local_trends:
                if trend["segment"] < valley_position * self.n_segments:
                    if trend["direction"] == "down":
                        has_down_before = True
                else:
                    if trend["direction"] == "up":
                        has_up_after = True
            
            if has_down_before and has_up_after:
                has_valley = True
        
        consecutive_down = 0
        for trend in reversed(local_trends):
            if trend["direction"] == "down":
                consecutive_down += 1
            else:
                break
        
        late_down_problem = consecutive_down > self.n_segments // 3
        
        return (global_direction, slope_global, has_valley, valley_depth, 
                valley_position, local_trends, late_down_problem)
    
    def _simple_trend_analysis(self, times, flux_I):
        n = len(flux_I)
        
        if n < 5:
            return "stable", 0, False, 0, 0, [], False
        
        start_time = times[0]
        time_seconds = np.array([(t - start_time).total_seconds() for t in times])
        
        slope, _, _, p, _ = stats.linregress(time_seconds, flux_I)
        
        if p < self.trend_significance and abs(slope) > self.slope_threshold:
            direction = "down" if slope < 0 else "up"
        else:
            direction = "stable"
        
        min_idx = np.argmin(flux_I)
        min_value = flux_I[min_idx]
        start_value = flux_I[0]
        
        has_valley = False
        valley_depth = 0
        valley_position = min_idx / n
        
        if start_value > 0:
            valley_depth = (start_value - min_value) / start_value * 100
        
        if valley_depth > self.valley_depth_threshold and 0.2 < valley_position < 0.8:
            has_valley = True
        
        return direction, slope, has_valley, valley_depth, valley_position, [], False
    
    def _check_single_frequency(self, date: datetime.date, array: str, frequency: int) -> GratingCheckResult:

        
        try:
            corr = srhcp.SRHCorrPlot(date, array, frequency, "corrplot_cache")
            
            if corr.data is None:
                return GratingCheckResult(
                    frequency=frequency,
                    state=DataStatus.NO_DATA,
                    comment="Нет данных в FITS файле"
                )
            
            time_indices = [
                i for i, t in enumerate(corr.times) 
                if self.start_hour <= t.hour < self.end_hour
            ]
            
            if time_indices:
                time_start = corr.times[time_indices[0]].strftime('%H:%M')
                time_end = corr.times[time_indices[-1]].strftime('%H:%M')
                time_range_str = f"{time_start}-{time_end}"
            else:
                time_range_str = f"{self.start_hour}:00-{self.end_hour}:00"
            
            if len(time_indices) < 50:
                return GratingCheckResult(
                    frequency=frequency,
                    state=DataStatus.NO_DATA,
                    comment=f"Маловато точек: {len(time_indices)}",
                    n_points=len(time_indices),
                    time_range=time_range_str
                )
            
            flux_I = corr.flux_I[time_indices]
            times = [corr.times[i] for i in time_indices]
            
            flux_I = np.nan_to_num(flux_I, nan=0.0)
            
            flux_I_median = np.median(flux_I)
            flux_I_std = np.std(flux_I)
            
            if np.max(flux_I) > self.outlier_threshold:
                return GratingCheckResult(
                    frequency=frequency,
                    state=DataStatus.BAD,
                    comment=f"Аномальные выбросы: flux_I_max={np.max(flux_I):.1e}",
                    flux_I_median=flux_I_median,
                    flux_I_std=flux_I_std,
                    n_points=len(time_indices),
                    time_range=time_range_str
                )
            
            (global_direction, global_slope, has_valley, valley_depth, 
             valley_position, local_trends, late_down_problem) = self._detect_local_trends(times, flux_I)
            
            local_trends_summary = []
            for t in local_trends:
                local_trends_summary.append({
                    "segment": t["segment"],
                    "direction": t["direction"]
                })
            
            is_problem = False
            problem_reason = []
            
            if global_direction == "down":
                is_problem = True
                problem_reason.append(f"глобальный нисходящий тренд (slope={global_slope:.3f})")
            
            if has_valley:
                is_problem = True
                problem_reason.append(f"галочка глубиной {valley_depth:.1f}% на позиции {valley_position:.0%}")
            
            if late_down_problem:
                is_problem = True
                down_segments = sum(1 for t in local_trends if t["direction"] == "down")
                problem_reason.append(f"{down_segments} из {len(local_trends)} сегментов нисходящие")
            
            if is_problem:
                return GratingCheckResult(
                    frequency=frequency,
                    state=DataStatus.PROBLEM,
                    comment=f"Проблемный тренд: {', '.join(problem_reason)}",
                    flux_I_median=flux_I_median,
                    flux_I_std=flux_I_std,
                    trend_slope=global_slope,
                    trend_direction=global_direction,
                    has_valley=has_valley,
                    valley_depth_pct=valley_depth,
                    valley_position=valley_position,
                    local_trends=local_trends_summary,
                    n_points=len(time_indices),
                    time_range=time_range_str
                )
            
            if global_direction == "up":
                comment = f"flux_I={flux_I_median:.1f} SFU, восходящий тренд"
            else:
                comment = f"flux_I={flux_I_median:.1f} SFU, стабильный"
            
            return GratingCheckResult(
                frequency=frequency,
                state=DataStatus.GOOD,
                comment=comment,
                flux_I_median=flux_I_median,
                flux_I_std=flux_I_std,
                trend_slope=global_slope,
                trend_direction=global_direction,
                has_valley=False,
                local_trends=local_trends_summary,
                n_points=len(time_indices),
                time_range=time_range_str
            )
            
        except Exception as e:
            return GratingCheckResult(
                frequency=frequency,
                state=DataStatus.NO_DATA,
                comment=f"Ошибка: {str(e)}"
            )
    
    def check_day(self, date: datetime.date, array: str) -> Tuple[Dict[int, GratingCheckResult], List[int]]:
        
        try:
            test_corr = srhcp.SRHCorrPlot(date, array, 6000, "corrplot_cache")
            if test_corr.data is not None and hasattr(test_corr, 'frequencies'):
                all_frequencies = sorted(test_corr.frequencies.tolist())
            else:
                all_frequencies = []
        except:
            all_frequencies = []
        
        if not all_frequencies:
            all_frequencies = [3000, 4000, 5000, 6000, 7000, 8000]
        
        results = {}
        for freq in all_frequencies:
            print(f"    Частота {freq} МГц...", end=" ")
            result = self._check_single_frequency(date, array, freq)
            results[freq] = result
            
            if result.state == DataStatus.GOOD:
                if result.trend_direction == "up":
                    print(f"✅ GOOD (восходящий)")
                else:
                    print(f"✅ GOOD (стабильный)")
            elif result.state == DataStatus.PROBLEM:
                if result.has_valley:
                    print(f"⚠️ PROBLEM (галочка {result.valley_depth_pct:.1f}% на {result.valley_position:.0%})")
                elif result.trend_direction == "down":
                    print(f"⚠️ PROBLEM (нисходящий тренд)")
                else:
                    print(f"⚠️ PROBLEM (проблемный тренд)")
            elif result.state == DataStatus.BAD:
                print(f"❌ BAD (выбросы)")
            else:
                print(f"❓ {result.state.value}")
        
        return results, all_frequencies


class AvailabilityChecker:

    
    def __init__(self, start_hour: int = 0, end_hour: int = 10):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.gratings = ['SRH0612', 'SRH1224', 'SRH0306']
    
    def check_day(self, date: datetime.date) -> Dict[str, bool]:
        t1 = datetime.datetime.combine(date, datetime.time(self.start_hour, 0, 0))
        t2 = datetime.datetime.combine(date, datetime.time(self.end_hour, 0, 0))
        
        try:
            frequencies = srhimages.get_frequencies(t1, t2)
            result = {}
            for grating in self.gratings:
                result[grating] = (grating in frequencies and len(frequencies[grating]) > 0)
            return result
        except Exception as e:
            print(f"Ошибка для даты {date}: {e}")
            return {grating: False for grating in self.gratings}
    
    def check_period(self, start_date: datetime.date, end_date: datetime.date) -> Dict[datetime.date, Dict[str, bool]]:
        results = {}
        current = start_date
        
        while current <= end_date:
            results[current] = self.check_day(current)
            current += datetime.timedelta(days=1)
        
        return results


class DataQualityManager:
    
    def __init__(self, 
                 cache_dir: str = "corrplot_cache", 
                 start_hour: int = 1, 
                 end_hour: int = 9,
                 n_segments: int = 8,              
                 valley_depth_threshold: float = 15, 
                 slope_threshold: float = 0.01):   
        self.cache_dir = cache_dir
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.n_segments = n_segments
        self.availability_checker = AvailabilityChecker()
        self.quality_checker = QualityChecker(
            start_hour=start_hour,
            end_hour=end_hour,
            n_segments=n_segments,
            valley_depth_threshold=valley_depth_threshold,
            slope_threshold=slope_threshold
        )
        self.results: Dict[datetime.date, DayCheckResult] = {}
    
    def check_period(self, start_date: datetime.date, end_date: datetime.date) -> Dict[datetime.date, DayCheckResult]:
        
        print("="*70)
        print(f"ПРОВЕРКА ПЕРИОДА: {start_date} - {end_date}")
        print(f"ВРЕМЕННОЕ ОКНО: с {self.start_hour}:00 до {self.end_hour}:00")
        print(f"КОЛИЧЕСТВО СЕГМЕНТОВ: {self.n_segments}")
        print("="*70)
        print("\nПАРАМЕТРЫ ЧУВСТВИТЕЛЬНОСТИ:")
        print(f"  - Минимальная глубина галочки: {self.quality_checker.valley_depth_threshold}%")
        print(f"  - Минимальный наклон для тренда: {self.quality_checker.slope_threshold}")
        print("="*70)
        
        print("\nШАГ 1: ПРОВЕРКА НАЛИЧИЯ ДАННЫХ")
        print("-"*50)
        
        availability = self.availability_checker.check_period(start_date, end_date)
        
        print("\nШАГ 2: ДЕТАЛЬНАЯ ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
        print(f"Анализ по {self.n_segments} сегментам")

        
        results = {}
        gratings = ['SRH0612', 'SRH1224', 'SRH0306']
        
        for date, avail in availability.items():
            print(f"\n📅 {date}:")
            
            quality = {}
            all_frequencies = {}
            
            for grating in gratings:
                if avail.get(grating, False):
                    print(f"  🔍 {grating}:")
                    quality[grating], all_frequencies[grating] = self.quality_checker.check_day(date, grating)
                else:
                    print(f"  ⏭️ {grating}: нет данных")
                    quality[grating] = {}
                    all_frequencies[grating] = []
            
            results[date] = DayCheckResult(
                date=date,
                availability=avail,
                quality=quality,
                all_frequencies=all_frequencies
            )
        
        self.results = results
        return results
    
    def save_to_json(self, filename: str = "data_quality.json"):
        data = {date.isoformat(): result.to_dict() for date, result in self.results.items()}
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Результаты сохранены в {filename}")
    
    def load_from_json(self, filename: str = "data_quality.json"):
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results = {
            datetime.date.fromisoformat(date_str): DayCheckResult.from_dict(result_dict)
            for date_str, result_dict in data.items()
        }
        
        print(f"✅ Загружено {len(self.results)} дней из {filename}")
    
    def get_summary_table(self, agg_method: str = "worst") -> pd.DataFrame:
        rows = []
        
        for date, result in sorted(self.results.items()):
            row = {'Date': date}
            for grating in ['SRH0612', 'SRH1224', 'SRH0306']:
                status = result.get_summary(grating, agg_method)
                row[grating] = status.value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.set_index('Date')
        return df
    
    def get_detailed_frequency_table(self) -> pd.DataFrame:
        rows = []
        
        for date, result in sorted(self.results.items()):
            for grating, freq_results in result.quality.items():
                for freq, check in freq_results.items():
                    rows.append({
                        'Date': date,
                        'Grating': grating,
                        'Frequency': freq,
                        'Status': check.state.value,
                        'Comment': check.comment,
                        'TimeRange': check.time_range,
                        'n_points': check.n_points,
                        'flux_I_median': check.flux_I_median,
                        'flux_I_std': check.flux_I_std,
                        'trend_slope': check.trend_slope,
                        'trend_direction': check.trend_direction,
                        'has_valley': check.has_valley,
                        'valley_depth_pct': check.valley_depth_pct,
                        'valley_position': check.valley_position
                    })
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        print("\n" + "="*70)
        print(f"СВОДНАЯ ТАБЛИЦА КАЧЕСТВА ДАННЫХ")
        print(f"ВРЕМЕННОЕ ОКНО: с {self.start_hour}:00 до {self.end_hour}:00")
        print(f"КОЛИЧЕСТВО СЕГМЕНТОВ: {self.n_segments}")
        print(f"ПОРОГ ГЛУБИНЫ ГАЛОЧКИ: {self.quality_checker.valley_depth_threshold}%")
        print("="*70)
        
        df = self.get_summary_table()
        print(df.to_string())
        
        detailed_df = self.get_detailed_frequency_table()
        
        print("\n" + "="*70)
        print("СТАТИСТИКА ПО ЧАСТОТАМ")
        print("="*70)
        
        if not detailed_df.empty:
            freq_stats = detailed_df.groupby(['Frequency', 'Status']).size().unstack(fill_value=0)
            print(freq_stats)
        
        print("\n" + "="*70)
        print("ОБЩАЯ СТАТИСТИКА")
        print("="*70)
        
        if not detailed_df.empty:
            stats = detailed_df['Status'].value_counts()
            total = len(detailed_df)
            
            print(f"\nВсего проверок (день×решётка×частота): {total}")
            for status, count in stats.items():
                if status == 'GOOD':
                    print(f"  ✅ GOOD: {count} ({count/total*100:.1f}%)")
                elif status == 'PROBLEM':
                    print(f"  ⚠️ PROBLEM: {count} ({count/total*100:.1f}%)")
                elif status == 'BAD':
                    print(f"  ❌ BAD: {count} ({count/total*100:.1f}%)")
                else:
                    print(f"  ❓ {status}: {count} ({count/total*100:.1f}%)")
    
    def get_problem_days(self) -> List[datetime.date]:
        return [date for date, result in self.results.items() if result.has_problem()]
    
    def get_problem_frequencies(self, grating: str) -> Dict[datetime.date, List[int]]:
        problems = {}
        for date, result in self.results.items():
            bad_freqs = result.get_bad_frequencies(grating)
            if bad_freqs:
                problems[date] = bad_freqs
        return problems

if __name__ == "__main__":
    manager = DataQualityManager(
        start_hour=1, 
        end_hour=9,
        n_segments=8,                      
        valley_depth_threshold=20,         
        slope_threshold=0.0055               
    )
    
    start = datetime.date(2024, 5, 7)
    end = datetime.date(2024, 5, 12)
    
    results = manager.check_period(start, end)
    
    manager.save_to_json("data_quality_detailed.json")
    
    manager.print_summary()
    
    problem_days = manager.get_problem_days()
    if problem_days:
        print(f"\n⚠️ Проблемные дни: {problem_days}")