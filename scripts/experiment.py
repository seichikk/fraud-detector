import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import aio_pika
import httpx
import psutil
from sqlalchemy import select

from fraud_detector.core.config import Settings
from fraud_detector.db.engine import create_engine
from fraud_detector.db.tasks import tasks as tasks_table

SCENARIOS: dict[str, dict[str, Any]] = {
    "bounded": {
        "title": "очередь на 100, prefetch 5",
        "env": {"TASKS_QUEUE": "tasks_bounded", "QUEUE_MAX_LENGTH": "100"},
        "limit": None,
    },
    "unbounded": {
        "title": "очередь без лимита, prefetch 5",
        "env": {"TASKS_QUEUE": "tasks_unbounded", "QUEUE_MAX_LENGTH": "0"},
        "limit": 20000,
    },
    "unbounded_prefetch": {
        "title": "ни лимита, ни prefetch",
        "env": {
            "TASKS_QUEUE": "tasks_unbounded_prefetch",
            "QUEUE_MAX_LENGTH": "0",
            "PREFETCH_COUNT": "0",
        },
        "limit": 20000,
    },
}

MONITOR_FIELDS = [
    "elapsed",
    "workers",
    "ready",
    "unacked",
    "unacked_ui",
    "queue_memory_mb",
    "broker_memory_mb",
    "source_rss_mb",
    "workers_rss_mb",
    "in_flight",
    "processed",
    "published",
    "rejected",
    "pause_ms",
]

FINISHED = re.compile(r"источник закончил: опубликовано (\d+), отказов (\d+)")


def executable(name: str) -> str:
    found = shutil.which(name, path=str(Path(sys.executable).parent))
    if found is None:
        raise SystemExit(f"не нашёл {name} рядом с {sys.executable}, сначала сделайте uv sync")
    return found


def open_log(path: Path) -> Any:
    return path.open("wb")


def prepare_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


async def spawn(name: str, args: list[str], env: dict[str, str], log: Path) -> Any:
    handle = open_log(log)
    try:
        return await asyncio.create_subprocess_exec(
            executable(name),
            *args,
            env=env,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
        )
    finally:
        handle.close()


async def stop(process: Any, patience: float = 90.0) -> None:
    if process.returncode is None:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), patience)
    except TimeoutError:
        print(f"процесс {process.pid} не уложился в {patience:.0f} с, добиваю")
        process.kill()
        await process.wait()


def rss_mb(pid: int) -> float:
    try:
        root = psutil.Process(pid)
        family = [root, *root.children(recursive=True)]
        return round(sum(member.memory_info().rss for member in family) / 2**20, 1)
    except psutil.Error:
        return 0.0


def read_progress(path: Path) -> dict[str, float]:
    empty = {"published": 0.0, "rejected": 0.0, "pause_seconds": 0.0}
    if not path.exists():
        return empty
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in reversed(rows):
        try:
            return {key: float(row[key]) for key in empty}
        except KeyError, TypeError, ValueError:
            continue
    return empty


def read_published(log: Path) -> int | None:
    found = FINISHED.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(found[-1][0]) if found else None


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def management_url(rabbit_url: str) -> tuple[str, tuple[str, str]]:
    parts = urlsplit(rabbit_url)
    auth = (parts.username or "guest", parts.password or "guest")
    return f"http://{parts.hostname or 'localhost'}:15672", auth


async def reset_queue(rabbit_url: str, queue: str) -> None:
    connection = await aio_pika.connect(rabbit_url)
    async with connection:
        channel = await connection.channel()
        await channel.queue_delete(queue)
        await channel.queue_delete(f"{queue}.dlq")


async def depth(connection: Any, queue: str) -> int:
    try:
        channel = await connection.channel()
    except aio_pika.exceptions.AMQPError:
        return 0
    try:
        declared = await channel.declare_queue(queue, passive=True)
        return int(declared.declaration_result.message_count or 0)
    except aio_pika.exceptions.AMQPError:
        return 0
    finally:
        if not channel.is_closed:
            await channel.close()


async def drain_ids(rabbit_url: str, queue: str) -> list[str]:
    connection = await aio_pika.connect(rabbit_url)
    async with connection:
        expected = await depth(connection, queue)
        ids: list[str] = []
        if expected == 0:
            return ids
        channel = await connection.channel()
        declared = await channel.declare_queue(queue, passive=True)
        async with declared.iterator(no_ack=True) as messages:
            async for message in messages:
                ids.append(message.message_id or "")
                if len(ids) >= expected:
                    break
        return ids


async def finished_tasks(database_url: str, since: datetime) -> list[Any]:
    engine = create_engine(database_url)
    statement = select(
        tasks_table.c.task_id,
        tasks_table.c.created_at,
        tasks_table.c.started_at,
        tasks_table.c.finished_at,
    ).where(tasks_table.c.finished_at >= since)
    try:
        async with engine.connect() as connection:
            return list((await connection.execute(statement)).mappings().all())
    finally:
        await engine.dispose()


async def broker_view(
    http: httpx.AsyncClient, base: str, auth: tuple[str, str], queue: str
) -> dict[str, float]:
    vhost = quote("/", safe="")
    try:
        info = (await http.get(f"{base}/api/queues/{vhost}/{queue}", auth=auth)).json()
        nodes = (await http.get(f"{base}/api/nodes", auth=auth)).json()
    except httpx.HTTPError, ValueError:
        return {}
    return {
        "unacked_ui": info.get("messages_unacknowledged", 0),
        "queue_memory_mb": round(info.get("memory", 0) / 2**20, 2),
        "broker_memory_mb": round(sum(node.get("mem_used", 0) for node in nodes) / 2**20, 1),
    }


async def worker_stats(http: httpx.AsyncClient, port: int) -> dict[str, Any]:
    try:
        response = await http.get(f"http://127.0.0.1:{port}/stats", timeout=1.0)
        return dict(response.json()) if response.status_code == 200 else {}
    except httpx.HTTPError, ValueError:
        return {}


async def wait_ready(http: httpx.AsyncClient, port: int, process: Any) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise SystemExit(f"воркер на порту {port} упал при старте, загляните в его лог")
        if (await worker_stats(http, port)).get("model_version"):
            return
        await asyncio.sleep(0.5)
    raise SystemExit(f"воркер на порту {port} так и не загрузил модель")


async def run_scenario(args: argparse.Namespace) -> None:
    settings = Settings()
    scenario = SCENARIOS[args.scenario]
    queue = scenario["env"]["TASKS_QUEUE"]
    out = prepare_folder(args.out / args.scenario)
    env = {**os.environ, "LOG_LEVEL": "INFO", **scenario["env"]}
    base, auth = management_url(settings.rabbit_url)
    limit = args.limit if args.limit is not None else scenario["limit"]

    print(f"сценарий «{scenario['title']}»: чищу очередь {queue}")
    await reset_queue(settings.rabbit_url, queue)

    connection = await aio_pika.connect(settings.rabbit_url)
    async with connection, httpx.AsyncClient() as http:
        workers: list[tuple[int, Any]] = []
        events: list[dict[str, Any]] = []

        async def start_worker(number: int) -> Any:
            port = args.first_port + number - 1
            process = await spawn(
                "fraud-worker",
                [],
                {**env, "WORKER_PORT": str(port)},
                out / f"worker-{number}.log",
            )
            workers.append((port, process))
            return process

        first = await start_worker(1)
        await wait_ready(http, args.first_port, first)
        print("первый воркер загрузил модель, запускаю источник")

        source_args = ["--file", str(args.file), "--metrics", str(out / "source.csv")]
        source_args += ["--batch-id", f"{args.scenario}-{datetime.now(UTC):%Y%m%d%H%M%S}"]
        if limit is not None:
            source_args += ["--limit", str(limit)]
        source = await spawn("fraud-source", source_args, env, out / "source.log")

        wall_start = datetime.now(UTC)
        clock = time.monotonic()
        rows: list[dict[str, Any]] = []
        helpers: list[asyncio.Task[None]] = []

        async def second_worker_ready(process: Any, launched: float) -> None:
            await wait_ready(http, args.first_port + 1, process)
            ready_at = round(time.monotonic() - clock, 1)
            events.append({"elapsed": ready_at, "event": "второй воркер готов"})
            print(f"[{ready_at:6.1f} с] второй воркер готов через {ready_at - launched:.1f} с")

        tick = clock
        while (elapsed := time.monotonic() - clock) < args.duration:
            if args.second_worker_at and len(workers) == 1 and elapsed >= args.second_worker_at:
                launched = round(elapsed, 1)
                events.append({"elapsed": launched, "event": "запуск второго воркера"})
                print(f"[{launched:6.1f} с] подключаю второй воркер")
                process = await start_worker(2)
                helpers.append(asyncio.create_task(second_worker_ready(process, launched)))

            view = await broker_view(http, base, auth, queue)
            stats = [await worker_stats(http, port) for port, _ in workers]
            progress = read_progress(out / "source.csv")
            in_flight = sum(item.get("in_flight", 0) for item in stats)
            row = {
                "elapsed": round(elapsed, 2),
                "workers": len(workers),
                "ready": await depth(connection, queue),
                "unacked": in_flight,
                "unacked_ui": view.get("unacked_ui", 0),
                "queue_memory_mb": view.get("queue_memory_mb", 0),
                "broker_memory_mb": view.get("broker_memory_mb", 0),
                "source_rss_mb": rss_mb(source.pid) if source.returncode is None else 0.0,
                "workers_rss_mb": round(sum(rss_mb(process.pid) for _, process in workers), 1),
                "in_flight": in_flight,
                "processed": sum(item.get("processed", 0) for item in stats),
                "published": int(progress["published"]),
                "rejected": int(progress["rejected"]),
                "pause_ms": round(progress["pause_seconds"] * 1000, 1),
            }
            rows.append(row)
            if len(rows) % 10 == 0:
                print(
                    f"[{elapsed:6.1f} с] в очереди {row['ready']}, у воркеров {row['unacked']}, "
                    f"опубликовано {row['published']}, обработано {row['processed']}, "
                    f"пауза источника {row['pause_ms']} мс"
                )
            tick += 1
            await asyncio.sleep(max(0.0, tick - time.monotonic()))

        for helper in helpers:
            helper.cancel()

        print("время вышло, останавливаю источник")
        await stop(source)
        print("останавливаю воркеры, они доделывают то, что уже взяли")
        await asyncio.gather(*(stop(process) for _, process in workers))

    published = read_published(out / "source.log")
    left = await drain_ids(settings.rabbit_url, queue)
    finished = await finished_tasks(settings.database_url, wall_start)
    await reset_queue(settings.rabbit_url, queue)

    done_ids = {str(task["task_id"]) for task in finished}
    seen = done_ids | set(left)
    tasks = [
        {
            "finished": round((task["finished_at"] - wall_start).total_seconds(), 3),
            "latency": round((task["finished_at"] - task["created_at"]).total_seconds(), 3),
            "waiting": round((task["started_at"] - task["created_at"]).total_seconds(), 3),
        }
        for task in finished
    ]
    write_csv(out / "monitor.csv", MONITOR_FIELDS, rows)
    write_csv(out / "tasks.csv", ["finished", "latency", "waiting"], tasks)
    write_csv(out / "events.csv", ["elapsed", "event"], events)
    result = {
        "scenario": args.scenario,
        "title": scenario["title"],
        "duration": args.duration,
        "second_worker_at": args.second_worker_at,
        "published": published,
        "done": len(done_ids),
        "left_in_queue": len(left),
        "redelivered_after_done": len(done_ids & set(left)),
        "lost": None if published is None else published - len(seen),
    }
    write_json(out / "result.json", result)
    print(
        f"готово: опубликовано {published}, обработано {len(done_ids)}, "
        f"осталось в очереди {len(left)}, из них уже обработанных "
        f"{result['redelivered_after_done']}, потеряно {result['lost']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="эксперимент: очередь с ограничением и без")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="прогнать один сценарий")
    run.add_argument("scenario", choices=sorted(SCENARIOS))
    run.add_argument("--file", type=Path, default=Path("data/raw/transactions.csv"))
    run.add_argument("--duration", type=float, default=120.0)
    run.add_argument("--second-worker-at", type=float, default=60.0)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--first-port", type=int, default=8101)
    run.add_argument("--out", type=Path, default=Path("experiments"))

    plot = commands.add_parser("plot", help="нарисовать графики по готовым прогонам")
    plot.add_argument("--out", type=Path, default=Path("experiments"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "run":
        asyncio.run(run_scenario(args))
    else:
        from experiment_plots import draw_all

        draw_all(args.out, SCENARIOS)


if __name__ == "__main__":
    main()
