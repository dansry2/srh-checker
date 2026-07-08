#!/usr/bin/env python3
"""Интерфейс командной строки."""
import argparse
from datetime import date
from srh_data_report import check_day


def main():
    parser = argparse.ArgumentParser(
        description="Проверка качества данных СРХ за день"
    )
    parser.add_argument("check_date", help="Дата в формате YYYY-MM-DD")
    parser.add_argument("--to-json", help="Сохранить в JSON файл")
    parser.add_argument("--to-html", help="Сохранить в HTML файл")
    parser.add_argument("--to-md", help="Сохранить в Markdown файл")
    parser.add_argument("--to-pdf", help="Сохранить в PDF файл")
    parser.add_argument("--journal", help="Путь к Excel файлу журнала антенн")
    
    args = parser.parse_args()
    
    try:
        dt = date.fromisoformat(args.check_date)
    except ValueError:
        print(f"Ошибка: неверный формат даты '{args.check_date}'. Используйте YYYY-MM-DD")
        return
    
    result = check_day(dt, excel_path=args.journal)
    
    if args.to_json:
        result.to_json(args.to_json)
    elif args.to_html:
        result.to_html(args.to_html)
    elif args.to_md:
        result.to_markdown(args.to_md)
    elif args.to_pdf:
        result.to_pdf(args.to_pdf)
    else:
        print(result)


if __name__ == "__main__":
    main()