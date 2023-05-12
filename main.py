import asyncio
import datetime
import hashlib
import json
import os
import typing as t
from zoneinfo import ZoneInfo

import aiofiles
import aiohttp
import dictdiffer
import sentry_sdk

URL_WEBHOOK = os.environ["URL_WEBHOOK"]
USER_ID = os.environ["USER_ID"]
DOWNLOAD_FILES = bool(os.environ.get("DOWNLOAD_FILES"))


# fmt: off
DIFF_TYPE: t.TypeAlias = \
    tuple[t.Literal["add"],    t.Literal[""], list [tuple[str, str]]] \
  | tuple[t.Literal["change"], str,           tuple[str, str]       ] \
  | tuple[t.Literal["remove"], t.Literal[""], list [tuple[str, str]]]
# fmt: on


def get_current_time() -> datetime.datetime:
    return datetime.datetime.now(tz=ZoneInfo("Europe/Prague"))


def get_id_by_file_name(file_name: str) -> int:
    return int(
        file_name.removeprefix("2023_dalsi_kola_PR_skoly_")
        .removeprefix("hmp-")
        .removeprefix("soukrome-")
        .removesuffix(".pdf")
        .removesuffix(".xlsx")
    )


class RateLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._counter = 0

        asyncio.ensure_future(self._loop())

    async def __aenter__(self) -> None:
        while self._counter > self.limit:
            await asyncio.sleep(0.1)

        self._counter += 1

    async def __aexit__(self, *_, **__) -> None:
        pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            self._counter = 0


class DataGetter:
    def __init__(self) -> None:
        self._results: dict[str, str]
        self._rate_limiter = RateLimiter(10)

    async def get_data(self) -> dict[str, str]:
        self._results = {}
        names = self._generate_names()

        async with aiohttp.ClientSession() as session:
            self._session = session
            tasks = self._generate_tasks(names)

            done, _ = await asyncio.wait(tasks)

        for done_task in done:
            if done_task.exception() is not None:
                sentry_sdk.capture_exception(done_task.exception())

            self._handle_result(done_task.result(), done_task.get_name())

        return self._results

    def _generate_names(self) -> set[str]:
        names = set()
        for number in range(0, 99):
            names.add(f"2023_dalsi_kola_PR_skoly_hmp-{number}.pdf")
            names.add(f"2023_dalsi_kola_PR_skoly_hmp-{number}.xlsx")
            names.add(f"2023_dalsi_kola_PR_skoly_soukrome-{number}.pdf")
            names.add(f"2023_dalsi_kola_PR_skoly_soukrome-{number}.xlsx")

        return names

    def _generate_tasks(self, names) -> set[asyncio.Task[bytes]]:
        tasks = set()
        for name in names:
            tasks.add(asyncio.create_task(self._send_request(name), name=name))
        return tasks

    async def _send_request(self, name) -> bytes | t.Literal[False]:
        async with self._rate_limiter:
            async with self._session.get(
                "https://www.prahaskolska.eu/wp-content/uploads/2023/05/" + name,
                headers={"User-Agent": "2nd-circle-updates"},
            ) as response:
                if response.status == 404:
                    return False

                response.raise_for_status()
                await save_to_file(response, name)
                return await response.read()

    def _handle_result(self, file: bytes | t.Literal[False], name: str) -> None:
        if file is False:
            return

        self._results[name] = hashlib.sha256(file).hexdigest()


async def save_to_file(response: aiohttp.ClientResponse, name: str) -> None:
    if not DOWNLOAD_FILES:
        return

    is_private = "soukrome" in name
    is_excel = name.endswith(".xlsx")
    number = get_id_by_file_name(name)

    async with aiofiles.open(
        f"data/saves/{'private' if is_private else 'free'}/{'excel' if is_excel else 'pdf'}/{number}.{'xlsx' if is_excel else 'pdf'}",
        "wb",
    ) as file:
        async for chunk in response.content.iter_chunked(1024):
            await file.write(chunk)


async def report_to_discord(diff: DIFF_TYPE) -> None:
    async with aiohttp.ClientSession() as session:
        embeds = parse_embeds_to_report(diff)
        to_send = []
        for embed in embeds:
            to_send.append(embed)
            if len(to_send) < 10:
                continue

            await _send_report(session, to_send)

            to_send = []

        if len(to_send) > 0:
            await _send_report(session, to_send)


async def _send_report(
    session: aiohttp.ClientSession, embeds: list[dict[str, str]]
) -> None:
    async with session.post(
        URL_WEBHOOK,
        json={
            "content": f"<@{USER_ID}>",
            "embeds": embeds,
            "allowed_mentions": {"parse": ["roles", "users", "everyone"]},
        },
    ) as response:
        if response.status == 429:  # rate limited
            to_sleep = (int((await response.json())["retry_after"]) / 1000) + 0.15
            print(f"Discord rate limited me, waiting {to_sleep} seconds...")
            await asyncio.sleep(to_sleep)
            await _send_report(session, embeds)
            return

        response.raise_for_status()


def _build_embed(
    event_name: str,
    color: t.Literal["green"] | t.Literal["yellow"] | t.Literal["red"],
    file_name: str,
    file_hash: str | None,
    footer_text: str = "File hash: {file_hash}",
):
    return {
        "title": "File was " + event_name,
        "description": f"`{file_name}`",
        "url": "https://www.prahaskolska.eu/wp-content/uploads/2023/05/" + file_name,
        "color": {"green": 65280, "yellow": 16776960, "red": 16711680}[color],
        "footer": {"text": footer_text.format(file_hash=file_hash)},
        "timestamp": get_current_time().isoformat(),
    }


def parse_embeds_to_report(diff: DIFF_TYPE) -> list[dict[str, str]]:
    embeds: list[dict[str, str]] = []

    for type, key, value in diff:
        if type == "add":
            for file_name, file_hash in value:
                embeds.append(
                    _build_embed(
                        "added", "green", file_name=file_name, file_hash=file_hash
                    )
                )
        elif type == "change":
            key, (old_value, new_value) = value
            embeds.append(
                _build_embed(
                    "changed",
                    "yellow",
                    file_name=key,
                    file_hash=None,
                    footer_text=f"Old file hash: {old_value} -> New file hash: {new_value}",
                )
            )
        elif type == "remove":
            for file_name, file_hash in value:
                embeds.append(
                    _build_embed(
                        "removed", "red", file_name=file_name, file_hash=file_hash
                    )
                )
        else:
            try:
                raise RuntimeError(f"Unknown type: {type}")
            except RuntimeError as error:
                sentry_sdk.capture_exception(error)

    embeds.sort(
        key=lambda embed: get_id_by_file_name(embed["description"][1:-1])
        + (10000 if embed["description"][1:-1].endswith(".xlsx") else 0)
        + (1000 if "soukrome" in embed["description"] else 0)
    )
    return embeds


def prepare_folders() -> None:
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/latest.json"):
        with open("data/latest.json", "w") as data_file:
            json.dump({"date": get_current_time().isoformat()}, data_file)

    if DOWNLOAD_FILES:
        for folder_to_create in [
            "data/saves/free/pdf",
            "data/saves/free/excel",
            "data/saves/private/pdf",
            "data/saves/private/excel",
        ]:
            os.makedirs(folder_to_create, exist_ok=True)


async def main() -> None:
    sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=1.0)

    prepare_folders()
    with open("data/latest.json", "r") as data_file:
        previous_run = json.load(data_file)

    new_data = await DataGetter().get_data()
    new_data["date"] = get_current_time().isoformat()

    os.rename(
        "data/latest.json",
        "data/{}.json".format(
            datetime.datetime.fromisoformat(previous_run["date"]).strftime(
                "%Y %m %d_%H %M %S"
            )
        ),
    )
    with open("data/latest.json", "w") as data_file:
        json.dump(new_data, data_file)

    await report_to_discord(dictdiffer.diff(previous_run, new_data, ignore={"date"}))


if __name__ == "__main__":
    asyncio.run(main())
