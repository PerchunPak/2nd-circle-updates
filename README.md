# Second circle updates notifier

The official site where are posted schools for the second circle doesn't update the list of schools
often, so I created this script while was in the bus in hope, that they at least automatically
update files, so I can grab them.

If you want to be notified too, [here](https://discord.gg/c9W8ngUFMz) is the server used for this
script. You can also browse downloaded files [here](https://2nd-circle.perchun.it).

The script will be active untill end of all circles in 2023. Then you need to host it somehow.

## How to host

Just find a place where you can run it periodically, and set these enviroment variables:

- `URL_WEBHOOK` - URL of the Discord webhook, where to send notifications.
- `USER_ID` - your Discord user ID to ping.
- (optional) `SENTRY_DSN` - key for [Sentry](https://sentry.io). Needed to track the errors,
    because I do not look into logs anyway.
- (optional) `DOWNLOAD_FILES` - set to anything (example `1`) if you want to download all the found
    files.
- (debug) `LOG_LEVEL` - see [Loguru's docs](https://loguru.readthedocs.io/en/stable/api/logger.html#levels)
    for possible values. Default is `INFO` (20).
