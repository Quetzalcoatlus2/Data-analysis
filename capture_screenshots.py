"""Recapture README Weather analysis screenshots with normalized framing.

The goal is to keep the browser conditions consistent with the README note
(`1920x1080`, `100%` zoom, AMOLED theme) while framing the actual analysis
surface instead of the full browser chrome. That keeps the Weather captures
closer to the Life Expectancy screenshots in perceived density.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.async_api import ElementHandle, Page, TimeoutError, async_playwright


BASE_URL = "http://127.0.0.1:5000"
DATASET_PATH = Path("datasets/Project 1 - Weather Dataset.csv")
DISPLAY_NAME = DATASET_PATH.name
OUTPUT_DIR = Path("screenshots")
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
VISIBLE_ANALYSIS_HEIGHT = 1007

VIEW_SPECS: list[dict[str, Any]] = [
    {
        "view": "overview",
        "output": OUTPUT_DIR / "weather_overview_amoled.png",
        "wait_for": "main.container .panel",
        "mode": "element",
        "selector": "main.container .panel",
        "max_height": VISIBLE_ANALYSIS_HEIGHT,
    },
    {
        "view": "interactive",
        "output": OUTPUT_DIR / "weather_interactive_amoled.png",
        "wait_for": "main.container .panel",
        "mode": "element",
        "selector": "main.container .panel",
    },
    {
        "view": "forecast",
        "output": OUTPUT_DIR / "weather_detailed_amoled.png",
        "wait_for": "main.container .panel",
        "mode": "element",
        "selector": "main.container .panel",
        "max_height": VISIBLE_ANALYSIS_HEIGHT,
    },
    {
        "view": "correlation",
        "output": OUTPUT_DIR / "weather_correlation_amoled.png",
        "wait_for": "#corr",
        "mode": "ancestor",
        "selector": "#corr",
        "ancestor_levels": 1,
    },
    {
        "view": "categories",
        "output": OUTPUT_DIR / "weather_categories_amoled.png",
        "wait_for": ".category-chart-panel",
        "mode": "element",
        "selector": ".category-chart-panel",
    },
]


def _normalize_rect(raw_rect: dict[str, Any] | None) -> dict[str, int] | None:
    """Clamp a client rect to the current viewport before clipping."""
    if not raw_rect:
        return None

    x = max(0, int(round(float(raw_rect.get("x", 0)))))
    y = max(0, int(round(float(raw_rect.get("y", 0)))))
    width = max(1, int(round(float(raw_rect.get("width", 1)))))
    height = max(1, int(round(float(raw_rect.get("height", 1)))))

    width = min(width, VIEWPORT_WIDTH - x)
    height = min(height, VIEWPORT_HEIGHT - y)
    if width <= 0 or height <= 0:
        return None

    return {"x": x, "y": y, "width": width, "height": height}


def _fit_to_canvas(image_bytes: bytes) -> Image.Image:
    """Scale a screenshot to fill as much of a 1920x1080 canvas as possible."""
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = min(VIEWPORT_WIDTH / src.width, VIEWPORT_HEIGHT / src.height)
    target_size = (
        max(1, int(round(src.width * scale))),
        max(1, int(round(src.height * scale))),
    )
    resized = src.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (VIEWPORT_WIDTH, VIEWPORT_HEIGHT), (0, 0, 0))
    offset = (
        (VIEWPORT_WIDTH - resized.width) // 2,
        (VIEWPORT_HEIGHT - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


async def _upload_weather_dataset(page: Page) -> str:
    """Upload the Weather dataset and return the analyze URL without query params."""
    print("Uploading Weather dataset...")
    await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
    await page.evaluate("localStorage.setItem('theme', 'amoled')")
    await page.reload(wait_until="networkidle", timeout=60000)
    async with page.expect_navigation(wait_until="domcontentloaded", timeout=120000):
        await page.set_input_files("input[type='file']", str(DATASET_PATH))
        await page.click("button[type='submit']")
    await page.wait_for_load_state("networkidle", timeout=120000)
    await asyncio.sleep(2)
    return page.url.split("?")[0]


async def _wait_for_view(page: Page, selector: str) -> None:
    """Wait for a view anchor element, but continue on slow/best-effort cases."""
    try:
        await page.wait_for_selector(selector, state="attached", timeout=180000)
    except TimeoutError as exc:
        print(f"  WARNING: timed out waiting for {selector}: {exc}")


async def _resolve_element(page: Page, spec: dict[str, Any]) -> ElementHandle:
    """Resolve the target element or ancestor element for a view spec."""
    handle = await page.query_selector(spec["selector"])
    if handle is None:
        raise RuntimeError(f"Could not find capture selector: {spec['selector']}")

    if spec["mode"] == "element":
        return handle

    if spec["mode"] == "ancestor":
        levels = int(spec.get("ancestor_levels", 0))
        current: ElementHandle | None = handle
        for _ in range(levels):
            parent_handle = await current.evaluate_handle("node => node.parentElement")
            current = parent_handle.as_element()
            if current is None:
                raise RuntimeError(
                    f"Could not resolve ancestor level for selector: {spec['selector']}"
                )
        return current

    raise RuntimeError(f"Unsupported element resolution mode: {spec['mode']}")


async def _capture_element(page: Page, spec: dict[str, Any]) -> bytes:
    """Capture a region derived from an element bounding box."""
    element = await _resolve_element(page, spec)
    box = await element.evaluate(
        """node => {
            const r = node.getBoundingClientRect();
            return { x: r.x, y: r.y, width: r.width, height: r.height };
        }"""
    )
    rect = _normalize_rect(box)
    if not rect:
        raise RuntimeError("Could not resolve a capture rectangle for the current view.")
    if spec.get("max_height"):
        rect["height"] = min(rect["height"], int(spec["max_height"]))
    return await page.screenshot(clip=rect)


async def _capture_view(page: Page, analyze_base: str, spec: dict[str, Any]) -> None:
    url = (
        f"{analyze_base}?display={DISPLAY_NAME}"
        f"&view={spec['view']}"
        "&forecast_pct=0.05"
        "&contamination=0.02"
        "&data_range=1.0"
        "&selected_col="
    )
    print(f"Capturing {spec['view']}...")
    await page.goto(url, wait_until="domcontentloaded", timeout=180000)
    await page.wait_for_load_state("networkidle", timeout=180000)
    await _wait_for_view(page, spec["wait_for"])
    await asyncio.sleep(5)

    image_bytes = await _capture_element(page, spec)
    framed = _fit_to_canvas(image_bytes)
    framed.save(spec["output"])
    print(f"  DONE: saved {spec['output']}")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            device_scale_factor=1,
        )
        page = await context.new_page()

        analyze_base = await _upload_weather_dataset(page)
        for spec in VIEW_SPECS:
            await _capture_view(page, analyze_base, spec)

        await browser.close()
        print("")
        print("All Weather README analysis screenshots refreshed.")


if __name__ == "__main__":
    asyncio.run(main())
