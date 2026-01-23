import psycopg2
try:
    conn = psycopg2.connect(
        dbname="voiceagentdb",
        user="postgres",
        password="flask123",
        host="127.0.0.1",
        port="5432"
    )
    print("Connection successful with 127.0.0.1!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
