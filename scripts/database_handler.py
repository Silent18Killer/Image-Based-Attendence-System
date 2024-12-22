import csv

def fetch_attendance():
    with open("../data/attendance.csv", "r") as file:
        reader = csv.reader(file)
        return list(reader)
