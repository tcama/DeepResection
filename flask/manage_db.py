import sqlite3, csv, json


def init_db():
    connection = sqlite3.connect('database.db')

    with open('schema.sql') as f:
        connection.executescript(f.read())

    cur = connection.cursor()

    connection.commit()
    connection.close()
    
def show_db():
    connection = sqlite3.connect('database.db')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM feedback")
    print(cursor.fetchall())
