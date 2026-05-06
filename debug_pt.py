from curl_cffi import requests as curl_requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

for pt in ["Efficiency", "Scoring", "Possessions", "CatchShoot", "PullUpShot",
           "Defense", "Drives", "Passing", "ElbowTouch", "PostTouch",
           "PaintTouch", "SpeedDistance"]:
    url = "https://stats.nba.com/stats/leaguedashptstats"
    params = {
        "LeagueID": "00", "PerMode": "Totals",
        "Season": "2023-24", "SeasonType": "Regular Season",
        "PtMeasureType": pt, "DateFrom": "", "DateTo": "",
    }
    r = curl_requests.get(url, params=params, headers=HEADERS,
                          impersonate="chrome110", timeout=30)
    print(f"{pt:20s}  {r.status_code}")
