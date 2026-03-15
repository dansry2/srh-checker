import srhimages
import datetime
import pandas as pd
import os

def check_data_availability(start_date, end_date, start_hour=0, end_hour=10, output_file='data_check.csv'):
    gratings = ['SRH0612', 'SRH1224', 'SRH0306']
    
    current = start_date
    if isinstance(current, datetime.datetime):
        current = current.date()
    end = end_date.date() if isinstance(end_date, datetime.datetime) else end_date
    #fvbvkhdbvld
    dates = []
    while current <= end:
        dates.append(current)
        current += datetime.timedelta(days=1)
    
    if os.path.exists(output_file):
        print(f"Загружаем существующий файл: {output_file}")
        df = pd.read_csv(output_file)
        
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.set_index('Date')
        
        for grating in gratings:
            if grating not in df.columns:
                df[grating] = '-'
    else:
        print("Создаем новый файл")
        df = pd.DataFrame(index=dates, columns=gratings)
        df.index.name = 'Date'
    
    days_processed = 0
    for date in dates:
        if date in df.index and pd.notna(df.loc[date, gratings[0]]):
            continue
        
        t1 = datetime.datetime.combine(date, datetime.time(start_hour, 0, 0))
        t2 = datetime.datetime.combine(date, datetime.time(end_hour, 0, 0))
        
        try:
            frequencies = srhimages.get_frequencies(t1, t2)
            
            row_data = {'Date': date}
            for grating in gratings:
                if grating in frequencies and len(frequencies[grating]) > 0:
                    row_data[grating] = '+'
                else:
                    row_data[grating] = '-'
            
            if date in df.index:
                for grating in gratings:
                    df.loc[date, grating] = row_data[grating]
            else:
                df.loc[date] = [row_data[g] for g in gratings]
            df.to_csv(output_file)
            print(f"  Результат для {date} сохранен")
            days_processed += 1
            
        except Exception as e:
            print(f"Ошибка для даты {date}: {e}")
            if date in df.index:
                for grating in gratings:
                    df.loc[date, grating] = '-'
            else:
                df.loc[date] = ['-', '-', '-']
            df.to_csv(output_file)
    
    return df.sort_index()

start = datetime.date(2022, 4, 1)
end = datetime.date(2022, 5, 30)

result = check_data_availability(start, end, start_hour=0, end_hour=10)

