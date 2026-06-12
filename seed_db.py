# -*- coding: utf-8 -*-
"""
Ixtiyoriy: DB ni demo ma'lumotlar bilan to'ldirish (grafiklarni sinash uchun).
Sizning asl skriptingiz asosida — jadval sxemasi app.py bilan mos (line_name qo'shilgan).
Real ma'lumotlarni O'CHIRMAYDI, faqat qo'shadi.

Ishga tushirish:  python seed_db.py
"""
import sqlite3
import random
from datetime import datetime, timedelta

conn = sqlite3.connect("street_crossing.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    type TEXT,
    direction TEXT,
    line_name TEXT
)
""")

types = ["person", "car"]
directions = ["N->S", "S->N", "E->W", "W->E"]
start_date = datetime.now() - timedelta(days=14)
days = 14

data = []
for day in range(days):
    for hour in range(24):
        current_time = start_date + timedelta(days=day, hours=hour)
        # tunda kam, kunduzi ko'p — realistik
        factor = 0.2 if hour < 6 or hour > 22 else 1.0
        people_count = int(random.randint(5, 60) * factor)
        car_count = int(random.randint(2, 25) * factor)
        for _ in range(people_count):
            data.append((current_time.strftime("%Y-%m-%d %H:%M:%S"),
                         "person", random.choice(directions), "Demo-liniya"))
        for _ in range(car_count):
            data.append((current_time.strftime("%Y-%m-%d %H:%M:%S"),
                         "car", random.choice(directions), "Demo-liniya"))

cursor.executemany(
    "INSERT INTO crossings (timestamp, type, direction, line_name) VALUES (?, ?, ?, ?)",
    data)
conn.commit()
print(f"Inserted rows: {len(data)}")
conn.close()
