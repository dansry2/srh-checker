"""Модуль с объектом результата проверки."""
import json
import os
from datetime import date
from typing import Dict, Any, Optional
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class CheckReport:
    """Единый объект результата проверки дня."""
    
    def __init__(self, check_date: date, raw_data: Dict[str, Any]):
        self.date = check_date
        self.raw_data = raw_data
    
    @property
    def summary(self) -> Dict[str, str]:
        result = {}
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating in self.raw_data:
                result[grating] = self._get_overall_status(self.raw_data[grating])
        return result
    
    def _get_overall_status(self, grating_data: dict) -> str:
        time_range = grating_data.get("time_range", "NO_DATA")
        flux = grating_data.get("flux", {})
        has_bad = any(f.get("state") == "BAD" for f in flux.values())
        has_problem = any(f.get("state") == "PROBLEM" for f in flux.values())
        if time_range == "BAD" or has_bad:
            return "BAD"
        elif time_range == "PROBLEM" or has_problem:
            return "PROBLEM"
        elif time_range == "GOOD":
            return "GOOD"
        return "NO_DATA"
    
    def _analyze_grating(self, grating_data: dict) -> dict:
        time_range = grating_data.get("time_range", "NO_DATA")
        flux_data = grating_data.get("flux", {})
        status_counts = defaultdict(int)
        all_freqs = []
        for freq_str, freq_info in flux_data.items():
            state = freq_info.get("state", "NO_DATA")
            status_counts[state] += 1
            all_freqs.append({
                "frequency": freq_str, "state": state,
                "comment": freq_info.get("comment", ""),
                "flux_median": freq_info.get("flux_I_median", 0),
                "flux_mean": freq_info.get("flux_I_mean", 0),
                "sfu_ratio": freq_info.get("sfu_ratio", 0),
                "expected_sfu": freq_info.get("expected_sfu", 0),
                "time_start": freq_info.get("time_start", ""),
                "time_range": freq_info.get("time_range", "")
            })
        all_freqs.sort(key=lambda x: int(x["frequency"]))
        return {
            "overall_status": self._get_overall_status(grating_data),
            "time_range": time_range, "status_counts": dict(status_counts),
            "all_freqs": all_freqs, "total_freqs": len(flux_data),
            "availability": grating_data.get("availability", False),
            "journal_notes": grating_data.get("journal_notes", {}).get("details", "")
        }
    
    def to_dict(self) -> dict:
        result = {"date": self.date.isoformat()}
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating not in self.raw_data:
                continue
            data = self.raw_data[grating]
            entry = {
                "availability": data.get("availability", False),
                "time_range": data.get("time_range", "NO_DATA"),
                "flux": {},
            }
            for freq, info in data.get("flux", {}).items():
                entry["flux"][freq] = {
                    "state": info.get("state", "NO_DATA"),
                    "comment": info.get("comment", ""),
                    "flux_I_median": info.get("flux_I_median", 0),
                    "flux_I_mean": info.get("flux_I_mean", 0),
                    "expected_sfu": info.get("expected_sfu", 0),
                    "sfu_ratio": info.get("sfu_ratio", 0),
                }
            journal = data.get("journal_notes", {})
            if isinstance(journal, dict) and journal.get("details"):
                entry["journal_notes"] = {"details": journal["details"]}
            elif isinstance(journal, str) and journal:
                entry["journal_notes"] = {"details": journal}
            result[grating] = entry
        return result
    
    def to_json(self, filepath=None):
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"JSON сохранён: {filepath}")
            return None
        return json_str
    
    def to_markdown(self, filepath=None):
        lines = [
            f"# Отчёт проверки за {self.date.isoformat()}", "",
            "## Сводка", "",
            "| Решётка | Статус | Доступность | Time Range | Частот | GOOD | PROBLEM | BAD |",
            "|---------|--------|-------------|------------|--------|-------|---------|-----|",
        ]
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating not in self.raw_data:
                continue
            a = self._analyze_grating(self.raw_data[grating])
            emoji = {"GOOD": "OK", "PROBLEM": "WARN", "BAD": "BAD", "NO_DATA": "N/A"}.get(a["overall_status"], "?")
            avail = "Yes" if a["availability"] else "No"
            lines.append(
                f"| {grating} | {emoji} {a['overall_status']} | {avail} | {a['time_range']} | "
                f"{a['total_freqs']} | {a['status_counts'].get('GOOD', 0)} | "
                f"{a['status_counts'].get('PROBLEM', 0)} | {a['status_counts'].get('BAD', 0)} |"
            )
        lines += ["", "---", ""]
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating not in self.raw_data:
                continue
            a = self._analyze_grating(self.raw_data[grating])
            lines.append(f"## {grating}\n")
            if a["journal_notes"]:
                lines.append(f"> Journal: {a['journal_notes']}\n")
            if a["all_freqs"]:
                lines.append("| Freq | State | Median | Expected | Ratio | Comment |")
                lines.append("|------|-------|--------|----------|-------|---------|")
                for f in a["all_freqs"]:
                    lines.append(
                        f"| {f['frequency']} | {f['state']} | {f['flux_median']:.1f} | "
                        f"{f['expected_sfu']:.1f} | {f['sfu_ratio']:.2f} | {f['comment'][:50]} |"
                    )
            else:
                lines.append("No frequency data")
            lines.append("")
        md_str = "\n".join(lines)
        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_str)
            print(f"Markdown saved: {filepath}")
            return None
        return md_str
    
    def to_html(self, filepath=None):
        parts = [f"""<!DOCTYPE html><html lang="ru">
<head><meta charset="utf-8"><title>Report {self.date.isoformat()}</title>
<style>body{{font-family:Arial;margin:20px}}table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:center}}th{{background:#666;color:white}}
.good{{color:green;font-weight:bold}}.problem{{color:orange;font-weight:bold}}.bad{{color:red;font-weight:bold}}
.journal{{background:#fffde7;padding:10px;border-left:4px solid #ffc107;margin:10px 0}}
h2{{border-bottom:2px solid #eee;padding-bottom:5px}}</style></head>
<body><h1>Report {self.date.isoformat()}</h1><h2>Summary</h2>
<table><tr><th>Grating</th><th>Status</th><th>Avail</th><th>Time Range</th><th>Freqs</th><th>GOOD</th><th>PROBLEM</th><th>BAD</th></tr>"""]
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating not in self.raw_data:
                continue
            a = self._analyze_grating(self.raw_data[grating])
            cls = {"GOOD":"good","PROBLEM":"problem","BAD":"bad"}.get(a["overall_status"],"")
            parts.append(
                f"<tr><td><b>{grating}</b></td><td class='{cls}'>{a['overall_status']}</td>"
                f"<td>{'Yes' if a['availability'] else 'No'}</td><td>{a['time_range']}</td>"
                f"<td>{a['total_freqs']}</td><td>{a['status_counts'].get('GOOD',0)}</td>"
                f"<td>{a['status_counts'].get('PROBLEM',0)}</td><td>{a['status_counts'].get('BAD',0)}</td></tr>")
        parts.append("</table>")
        for grating in ['SRH0306', 'SRH0612', 'SRH1224']:
            if grating not in self.raw_data:
                continue
            a = self._analyze_grating(self.raw_data[grating])
            parts.append(f"<h2>{grating}</h2>")
            if a["journal_notes"]:
                parts.append(f"<div class='journal'><b>Journal:</b> {a['journal_notes']}</div>")
            if a["all_freqs"]:
                parts.append("<table><tr><th>Freq</th><th>State</th><th>Median</th><th>Expected</th><th>Ratio</th><th>Comment</th></tr>")
                for f in a["all_freqs"]:
                    cls = {"GOOD":"good","PROBLEM":"problem","BAD":"bad"}.get(f["state"],"")
                    parts.append(
                        f"<tr><td>{f['frequency']}</td><td class='{cls}'>{f['state']}</td>"
                        f"<td>{f['flux_median']:.1f}</td><td>{f['expected_sfu']:.1f}</td>"
                        f"<td>{f['sfu_ratio']:.2f}</td><td>{f['comment'][:50]}</td></tr>")
                parts.append("</table>")
            else:
                parts.append("<p>No data</p>")
        parts.append("</body></html>")
        html = "\n".join(parts)
        if filepath:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML saved: {filepath}")
            return None
        return html
    
    def to_pdf(self, filepath, grating=None):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        gratings = [grating] if grating else ['SRH0306','SRH0612','SRH1224']
        with PdfPages(filepath) as pdf:
            for gr in gratings:
                if gr not in self.raw_data:
                    continue
                a = self._analyze_grating(self.raw_data[gr])
                fig, ax = plt.subplots(figsize=(12,10))
                ax.axis('off')
                y = 0.95
                ax.text(0.05, y, 'Data Quality Report', fontsize=18, fontweight='bold', transform=ax.transAxes)
                y -= 0.05
                ax.text(0.05, y, f'Date: {self.date.isoformat()}    Grating: {gr}', fontsize=14, transform=ax.transAxes)
                y -= 0.06
                ax.text(0.05, y, 'Summary:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                y -= 0.03
                for line in [
                    f"Status: {a['overall_status']}", f"Available: {'Yes' if a['availability'] else 'No'}",
                    f"Time Range: {a['time_range']}", f"Frequencies: {a['total_freqs']}",
                    f"GOOD: {a['status_counts'].get('GOOD',0)} | PROBLEM: {a['status_counts'].get('PROBLEM',0)} | BAD: {a['status_counts'].get('BAD',0)}"
                ]:
                    ax.text(0.08, y, line, fontsize=11, transform=ax.transAxes)
                    y -= 0.025
                y -= 0.01
                if a["journal_notes"]:
                    notes = self._wrap_text(a["journal_notes"])
                    if notes:
                        if y - len(notes)*0.025 < 0.1:
                            pdf.savefig(fig); plt.close(fig)
                            fig, ax = plt.subplots(figsize=(12,10)); ax.axis('off'); y = 0.95
                        ax.text(0.05, y, 'Journal:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                        y -= 0.03
                        for line in notes:
                            ax.text(0.08, y, line, fontsize=10, transform=ax.transAxes,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
                            y -= 0.025
                        y -= 0.02
                ax.text(0.05, y, 'Frequencies:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                y -= 0.03
                for f in a["all_freqs"]:
                    if y < 0.05:
                        pdf.savefig(fig); plt.close(fig)
                        fig, ax = plt.subplots(figsize=(12,10)); ax.axis('off'); y = 0.95
                    ax.text(0.08, y, f"{f['frequency']}: {f['state']} (median={f['flux_median']:.1f}, SFU={f['expected_sfu']:.1f})",
                            fontsize=9, transform=ax.transAxes)
                    y -= 0.022
                pdf.savefig(fig); plt.close(fig)
        print(f"PDF saved: {filepath}")
    
    def _wrap_text(self, text, max_chars=80):
        if not text: return []
        words = text.split()
        lines, cur = [], ""
        for w in words:
            t = cur + (" " if cur else "") + w
            if len(t) <= max_chars: cur = t
            else:
                if cur: lines.append(cur)
                cur = w[:max_chars]
        if cur: lines.append(cur)
        return lines
    
    def __repr__(self):
        parts = [f"CheckReport({self.date.isoformat()})"]
        for g in ['SRH0306','SRH0612','SRH1224']:
            if g in self.raw_data:
                a = self._analyze_grating(self.raw_data[g])
                parts.append(f"  {g}: {a['overall_status']} ({a['total_freqs']} freq)")
        return "\n".join(parts)