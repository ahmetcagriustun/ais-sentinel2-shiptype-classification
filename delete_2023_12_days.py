from datetime import datetime, timedelta

from utils.config_utils import load_config
from utils.db_utils import create_pg_connection

# Days to delete (inclusive), in YYYY-MM-DD format
DAYS = [
    "2023-12-01",
    "2023-12-19",
    "2023-12-22",
    "2023-12-23",
    "2023-12-24",
    "2023-12-25",
    "2023-12-27",
    "2023-12-30",
    "2023-12-31"
]


def main():
    # Load DB config and open connection
    cfg = load_config("config.yaml")
    db_conf = cfg["database"]
    conn = create_pg_connection(db_conf)

    try:
        with conn.cursor() as cur:
            for d_str in DAYS:
                d = datetime.strptime(d_str, "%Y-%m-%d")
                d_next = d + timedelta(days=1)

                print(f"[delete] Deleting rows for {d_str} ...")
                cur.execute(
                    """
                    DELETE FROM public.ship_raw_data
                    WHERE "# Timestamp" >= %s
                      AND "# Timestamp" <  %s;
                    """,
                    (d, d_next),
                )
                print(f"[delete] {d_str}: deleted {cur.rowcount} rows")
                conn.commit()

        print("[delete] Done. All selected days processed.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
