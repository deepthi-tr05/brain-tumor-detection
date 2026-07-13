import sqlite3
conn = sqlite3.connect('users.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])
users = c.execute('SELECT id, username, email FROM users LIMIT 5').fetchall()
print('Users:', [dict(u) for u in users])
conn.close()
