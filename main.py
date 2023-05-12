import asyncio
import datetime
import hashlib
import json
import os
import typing as t

import aiohttp
import dictdiffer
import sentry_sdk

URL_WEBHOOK = os.environ["URL_WEBHOOK"]
USER_ID = os.environ["USER_ID"]


# fmt: off
DIFF_TYPE: t.TypeAlias = \
    tuple[t.Literal["add"],    t.Literal[""], list [tuple[str, str]]] \
  | tuple[t.Literal["change"], str,           tuple[str, str]       ] \
  | tuple[t.Literal["remove"], t.Literal[""], list [tuple[str, str]]]
# fmt: on


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

    async def _send_request(self, name) -> bytes:
        async with self._rate_limiter:
            async with self._session.get(
                "https://www.prahaskolska.eu/wp-content/uploads/2023/05/" + name
            ) as response:
                if response.status not in {200, 404}:
                    raise RuntimeError(f"Unknown status code: {response.status}")

                return await response.read()

    def _handle_result(self, file: bytes, name: str) -> None:
        self._results[name] = hashlib.sha256(file).hexdigest()


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


async def _send_report(session: aiohttp.ClientSession, embeds: list[dict[str, str]]) -> None:
    async with session.post(
        URL_WEBHOOK,
        json={
            "content": f"<@{USER_ID}>",
            "embeds": embeds,
            "allowed_mentions": {"parse": ["roles", "users", "everyone"]},
        },
    ) as response:
        if response.status != 200:
            raise RuntimeError("Failed to report")


def _build_embed(
    event_name: str, color: t.Literal["green"] | t.Literal["yellow"] | t.Literal["red"], file_name: str, file_hash: str | None, footer_text:str="File hash: {file_hash}"
):
    return {
        "title": "File was " + event_name,
        "description": f"`{file_name}`",
        "url": "https://www.prahaskolska.eu/wp-content/uploads/2023/05/" + file_name,
        "color": {"green": 65280, "yellow": 16776960, "red": 16711680}[color],
        "footer": {"text": footer_text.format(file_hash=file_hash)},
    }


def parse_embeds_to_report(diff: DIFF_TYPE) -> list[dict[str, str]]:
    embeds = []

    for type, key, value in diff:
        if type == "add":
            key, value = value
            embeds.append(
                _build_embed("added", "green", file_name=key, file_hash=value)
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
            key, value = value
            embeds.append(
                _build_embed("removed", "red", file_name=key, file_hash=value)
            )
        else:
            try:
                raise RuntimeError(f"Unknown type: {type}")
            except RuntimeError as error:
                sentry_sdk.capture_exception(error)

    return embeds


async def main() -> None:
    sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=1.0)

    with open("data/latest.json", "r") as data_file:
        previous_run = json.load(data_file)

    new_data = await DataGetter().get_data()
    new_data["date"] = datetime.datetime.now().isoformat()

    await report_to_discord(dictdiffer.diff(previous_run, new_data, ignore={"date"}))

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


if __name__ == "__main__":
    asyncio.run(main())
