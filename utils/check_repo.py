import requests
from datetime import datetime

OWNER = "nodchip"
REPO = "nnue-pytorch"
GITHUB_API = f"https://api.github.com/repos/{OWNER}/{REPO}/forks"

# 任意：必要ならトークンを入れる（レートリミット緩和用）
GITHUB_TOKEN = None  # 例: "ghp_xxx..."

def fetch_forks(page=1, per_page=100):
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    params = {
        "page": page,
        "per_page": per_page,
        "sort": "newest",  # API 上のソートキー。細かい並べ替えは後でやる
    }

    resp = requests.get(GITHUB_API, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def list_forks_sorted_by_pushed_at(max_pages=3):
    forks = []
    for page in range(1, max_pages + 1):
        data = fetch_forks(page=page)
        if not data:
            break
        forks.extend(data)

    # pushed_at の新しい順にソート
    forks.sort(key=lambda r: r.get("pushed_at", ""), reverse=True)

    print(f"Found {len(forks)} forks.")
    for repo in forks:
        name = repo["full_name"]
        url = repo["html_url"]
        pushed_at = repo.get("pushed_at")
        pushed_at_str = pushed_at or "N/A"
        if pushed_at:
            dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            pushed_at_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"- {name:40s}  last pushed: {pushed_at_str}  ({url})")

if __name__ == "__main__":
    list_forks_sorted_by_pushed_at()
