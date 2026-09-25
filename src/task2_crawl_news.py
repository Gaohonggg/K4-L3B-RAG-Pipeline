"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://visithue.vn/3-ngay-2-dem-o-Hue.html/?pid=MTk4NzN8Y3NkbGRs0",
    "https://visithue.vn/chuyen-muc/diem-den/?id=NjF8Y3NkbGRs0",
    "https://visithue.vn/vi-the-dac-biet-cua-am-thuc-hue/?pid=MjI2Njd8Y3NkbGRs0",
    "https://visithue.vn/chuyen-muc/tour-du-lich/?id=MTE1MXxjc2RsZGw1",
    "https://visithue.vn/cau-truong-tien-diem-den-khong-the-bo-qua-khi-den-hue/?pid=MjMyNTJ8Y3NkbGRs0"
]


async def crawl_article(url: str) -> dict:
    from datetime import datetime
    from crawl4ai import AsyncWebCrawler
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # Xử lý metadata an toàn
        metadata = result.metadata or {}
        title = metadata.get("title", "Unknown") if isinstance(metadata, dict) else "Unknown"
        
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
