import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

BUCKET = os.environ.get("MLFLOW_ARTIFACTS_BUCKET", "mlflow-artifacts")
ATTEMPTS = 30


def main() -> None:
    s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
    for attempt in range(1, ATTEMPTS + 1):
        try:
            s3.head_bucket(Bucket=BUCKET)
        except ClientError:
            s3.create_bucket(Bucket=BUCKET)
            print(f"создан бакет {BUCKET}")
            return
        except BotoCoreError as exc:
            if attempt == ATTEMPTS:
                raise
            print(f"minio ещё не отвечает ({exc}), попытка {attempt} из {ATTEMPTS}")
            time.sleep(1)
        else:
            print(f"бакет {BUCKET} уже есть")
            return


if __name__ == "__main__":
    main()
