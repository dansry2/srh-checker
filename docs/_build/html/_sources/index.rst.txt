Документация srh-data-reports
===============================

Пакет для автоматической проверки качества данных **Сибирского Радиогелиографа (СРГ)**.

.. toctree::
   :maxdepth: 2
   :caption: Содержание:

   modules

Быстрый старт
-------------

Установка:

.. code-block:: bash

   pip install srh-data-reports

Командная строка:

.. code-block:: bash

   srh-data-report "2024-05-15"
   srh-data-report "2024-05-15" --to-json report.json
   srh-data-report "2024-05-15" --to-pdf report.pdf

Python:

.. code-block:: python

   from datetime import date
   from srh_data_report import check_day

   result = check_day(date(2024, 5, 15))
   print(result.summary)
   result.to_json("report.json")

Модули
------

.. toctree::
   :maxdepth: 1

   srh_data_report
