import os
from dotenv import load_dotenv
load_dotenv()

def get_conn():
    driver = os.getenv("DB_DRIVER", "mariadb").lower()

    if driver == "mariadb":
        import mariadb
        return mariadb.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "ofx"),
            password=os.getenv("DB_PASSWORD", "ofxpw"),
            database=os.getenv("DB_NAME", "openflights"),
            autocommit=True,
        )
    else:
        import pymysql
        return pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "ofx"),
            password=os.getenv("DB_PASSWORD", "ofxpw"),
            database=os.getenv("DB_NAME", "openflights"),
            autocommit=True,
            cursorclass=pymysql.cursors.Cursor,
        )