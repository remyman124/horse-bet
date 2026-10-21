#!/usr/bin/env python3
"""Scrape live odds from bet.hkjc.com for 2026-05-06 ST races."""
from playwright.sync_api import sync_playwright
import time, json, re

BASE = "https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{raceNo}?b_cid=SPLDSPA_hkjc-home_MegaMenu"

def scrape_race(page, date, venue, race_no):
    url = BASE.format(date=date, venue=venue, raceNo=race_no)
    page.goto(url, timeout=30000)
    page.wait_for_load_state("networkidle", timeout=20000)
    time.sleep1() = 2
    
    content = page.content()
    print(f"  R{race_no}: {len(content)} chars", end="")
    
    # Look for JSON data embedded in the page
    odds_data = {}
    
    # Method 1: find wagers JSON in page
    m = re.search(r'wagers\s*[=:]\s*(\[.*?\]);', content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            print(f" (wagers JSON: {len(data)} entries)", end="")
        except:
            pass
    
    # Method 2: look for odds in data attributes
    sel = page.query_selector_all("[data-odds]")
    if sel:
        print(f" (data-odds elements: {len(sel)})", end="")
    
    # Method 3: try to find in-page JavaScript variables
    for pat in [
        r'var\s+odds\s*=\s*(\{.*?\});',
        r'window\.odds\s*=\s*(\{.*?\});',
        r'"winOdds"\s*:\s*([0-9.]+)',
    ]:
        m = re.search(pat, content)
        if m:
            print(f"  Found: {m.group(0)[:100]}")
            break
    
    # Method 4: try to evaluate page data
    try:
        result = page.evaluate("""
            () => {
                // Try to find horse/odds data
                const scripts = document.querySelectorAll('script');
                let found = null;
                for (const s of scripts) {
                    const t = s.textContent;
                    if (t.includes('winOdds') || t.includes('horseNo')) {
                        found = t.substring(0, 500);
                        break;
                    }
                }
                return {
                    url: window.location.href,
                    bodyLen: document.body.innerHTML.length,
                    sample: found || document.body.innerHTML.substring(0, 300)
                };
            }
        """)
        print(f"  evaluate: {result['bodyLen']} chars")
        print(f"  sample: {result['sample'][:200]}")
    except Exception as e:
        print(f"  evaluate error: {e}")
    
    return odds_data

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    page = context.new_page()
    
    for race_no in range(1, 10):
        scrape_race(page, "2026-05-06", "ST", race_no)
    
    browser.close()
