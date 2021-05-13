from pathlib import Path
import plac
import sqlite3
from sqlite3 import Error
from src import load_config

@plac.opt(arg="env_path", help="Path to .env file", type=Path)
def create_db(env_path: str = "./.env.dev"):
    """
    Create a sqlite database. If no argument is supplied will default to
    creating a dev db
    """

    # Load the configuration
    config = load_config(env_path)    
    mlflow_tracking_db = config["mlflow_tracking_db"]

    # Set the conn
    conn = None

    try:
        conn = sqlite3.connect(mlflow_tracking_db)
        print("Connection Established")

    except Error as e:
        print(e)

    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    plac.call(create_db)
