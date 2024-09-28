import pandas as pd

df1 = pd.read_csv('input_data/main1.csv')
df2 = pd.read_csv('input_data/main2.csv')
df3 = pd.read_csv('input_data/main3.csv')

all_df = pd.concat([df1, df2, df3], sort = False, axis = 0)

all_df['table'] = 0
all_df.loc[all_df['full_name'] != 0, 'table'] = 1
all_df.loc[all_df['first_name'] != 0, 'table'] = 2
all_df.loc[all_df['name'] != 0, 'table'] = 3

column_names = ['full_name','first_name', 'middle_name', 'last_name', 'name']

for name in column_names:
    if name in all_df.columns:
        # Удаление лишних пробелов
        all_df[name] = all_df[name].str.strip()
        # Удаление лишних символов
        all_df[name] = all_df[name].str.replace(r"[^а-яА-ЯёЁ\s]", '', regex=True)

all_df = all_df.fillna(0)

all_df.reset_index().to_csv('input_data/alltables.csv', index=False)

import dedupe
import csv

# Чтение данных из CSV-файла
with open('input_data/alltables.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = {i: row for i, row in enumerate(reader)}

fields = [
    {'field': 'full_name', 'type': 'String'},
    {'field': 'address', 'type': 'String'},
    {'field': 'phone', 'type': 'String'},
    {'field': 'email', 'type': 'String'},
    {'field': 'sex', 'type': 'String'},
    {'field': 'birthdate', 'type': 'String'},
    {'field': 'first_name', 'type': 'String'},
    {'field': 'middle_name', 'type': 'String'},
    {'field': 'last_name', 'type': 'String'},
    {'field': 'name', 'type': 'String'},
]

with open('dedupe_model.pickle', 'rb') as f:
    deduper = dedupe.StaticDedupe(f)

# Определение порога для автоматического объединения дубликатов
threshold = deduper.threshold(data, recall_weight=1)

# Нахождение дубликатов в новых данных
clustered_dupes = deduper.match(data, threshold)

# Вывод дубликатов
for cluster in clustered_dupes:
    print(cluster)