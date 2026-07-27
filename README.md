# srh-data-reports



Пакет для автоматической проверки качества данных **Сибирского Радиогелиографа (СРГ)**.

Проверяет доступность решёток, временные последовательности, соответствие потока ожидаемому уровню SFU, выявляет аномалии. Результат можно экспортировать в JSON, HTML, Markdown и PDF.



## Установка



`pip install srh-data-reports`



## Быстрый старт



### Командная строка



```
# Проверить день и вывести в терминал

srh-data-report "2024-05-15"



\# Сохранить в разных форматах

srh-data-report "2024-05-15" --to-json report.json

srh-data-report "2024-05-15" --to-html report.html

srh-data-report "2024-05-15" --to-md report.md

srh-data-report "2024-05-15" --to-pdf report.pdf



\# С подключением к API журнала наблюдений

srh-data-report "2026-07-25" --api-url http://localhost:8000 --to-pdf report.pdf



\# Или через переменную окружения

export SRH\_API\_URL=http://localhost:8000

srh-data-report "2026-07-25" --to-pdf report.pdf```



### Python



```python
from datetime import date
from srh\\\_data\\\_report import check\\\_day

# Проверить день
result = check\\\_day(date(2024, 5, 15))

# Краткая сводка
print(result.summary)
# {'SRH0306': 'GOOD', 'SRH0612': 'GOOD', 'SRH1224': 'GOOD'}

# Экспорт
result.to\\\_json("report.json")
result.to\\\_html("report.html")
result.to\\\_markdown("report.md")
result.to\\\_pdf("report.pdf")

# С подключением к API журнала наблюдений

result = check\_day(date(2026, 7, 25), api\_url="http://localhost:8000")

result.to\_pdf("report\_with\_notes.pdf")```

\---

## Возможности

* **Доступность решёток** — проверка наличия данных для SRH0306, SRH0612, SRH1224
* **Временная последовательность** — выявление скачков и больших промежутков
* **Качество потока** — сравнение с ожидаемым уровнем SFU
* **Поиск аномалий** — обнаружение провалов данных
* **Интеграция с журналом антенн — получение заметок о состоянии антенн и диапазонов через JSON API**
* **Экспорт** — JSON, HTML, Markdown, PDF

\---

## Зависимости

* pandas
* numpy
* scipy
* astropy
* srhimages
* matplotlib

\---

## Структура пакета

```
srh-checker/
└── srh/srh\\\_data\\\_report/
    ├── \\\_\\\_init\\\_\\\_.py      # check\\\_day()
    ├── report.py        # CheckReport (JSON, HTML, MD, PDF)
    ├── cli.py           # CLI интерфейс
    ├── checks.py        # Проверки (AvailabilityChecker, QualityChecker)
    ├── fetch\_api.py     # Получение данных из JSON API журнала наблюдений
    └── ...
```

