"""
OneMap SG client — geocoding and public-transport routing.

Two very different auth stories:
  * /commonapi/search  — public, no token. Used to turn a postal code or an
    address into coordinates.
  * /api/public/routingsvc/route — needs a bearer token, obtained either from
    ONEMAP_TOKEN directly or by logging in with ONEMAP_EMAIL/ONEMAP_PASSWORD.

Routing degrades gracefully: if no credentials are configured (or OneMap is
having a bad day) the caller gets `None` and falls back to a maps link, so
/meetpoint still works without a OneMap account.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from app.ai.config import get_onemap_settings

logger = logging.getLogger("student_claw.ai.onemap")

_SGT = ZoneInfo("Asia/Singapore")
_TIMEOUT = httpx.Timeout(15.0)

# Cached bearer token (OneMap tokens last ~3 days; refresh an hour early).
_token_cache: dict[str, float | str] = {}


@dataclass(frozen=True)
class Place:
    name: str
    address: str
    postal_code: str
    latitude: float
    longitude: float

    @property
    def maps_url(self) -> str:
        return f"https://www.google.com/maps/search/?api=1&query={self.latitude},{self.longitude}"


@dataclass
class RouteLeg:
    mode: str  # WALK | BUS | SUBWAY | RAIL
    line: str  # bus service no. / MRT line, when known
    from_name: str
    to_name: str
    minutes: int


@dataclass
class Route:
    total_minutes: int
    walk_minutes: int
    transfers: int
    fare: Optional[str]
    legs: list[RouteLeg]


# ---------------------------------------------------------------------------
# Geocoding (public endpoint)
# ---------------------------------------------------------------------------
async def geocode(query: str) -> Optional[Place]:
    """
    Resolve a postal code or address to a Place. Returns None when OneMap finds
    nothing — the caller should ask the user to be more specific.
    """
    query = (query or "").strip()
    if not query:
        return None

    cfg = get_onemap_settings()
    url = f"{cfg.base_url}/api/common/elastic/search"
    params = {"searchVal": query, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("OneMap geocode failed for %r: %s", query, exc)
        return None

    results = data.get("results") or []
    if not results:
        return None

    top = results[0]
    try:
        lat = float(top.get("LATITUDE"))
        lon = float(top.get("LONGITUDE"))
    except (TypeError, ValueError):
        return None

    building = (top.get("BUILDING") or "").strip()
    road = (top.get("ROAD_NAME") or "").strip()
    name = building if building and building != "NIL" else (road or query)
    return Place(
        name=name,
        address=(top.get("ADDRESS") or "").strip(),
        postal_code=(top.get("POSTAL") or "").strip(),
        latitude=lat,
        longitude=lon,
    )


# ---------------------------------------------------------------------------
# Auth for the routing endpoint
# ---------------------------------------------------------------------------
async def _get_token() -> Optional[str]:
    """Return a valid bearer token, logging in if needed. None when unconfigured."""
    cfg = get_onemap_settings()
    if cfg.token:
        return cfg.token
    if not (cfg.email and cfg.password):
        return None

    cached = _token_cache.get("token")
    expiry = float(_token_cache.get("expiry") or 0)
    if cached and time.time() < expiry:
        return str(cached)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{cfg.base_url}/api/auth/post/getToken",
                json={"email": cfg.email, "password": cfg.password},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("OneMap token request failed: %s", exc)
        return None

    token = data.get("access_token")
    if not token:
        return None
    # Refresh an hour before the stated expiry.
    try:
        expiry_ts = float(data.get("expiry_timestamp", 0)) - 3600
    except (TypeError, ValueError):
        expiry_ts = time.time() + 3600
    _token_cache["token"] = token
    _token_cache["expiry"] = expiry_ts
    return token


def routing_available() -> bool:
    cfg = get_onemap_settings()
    return bool(cfg.token or (cfg.email and cfg.password))


# ---------------------------------------------------------------------------
# Public-transport routing
# ---------------------------------------------------------------------------
def _leg_from_raw(raw: dict) -> RouteLeg:
    mode = (raw.get("mode") or "").upper()
    route = (raw.get("route") or "").strip()
    seconds = raw.get("duration") or 0
    try:
        minutes = max(1, round(float(seconds) / 60))
    except (TypeError, ValueError):
        minutes = 1
    return RouteLeg(
        mode=mode,
        line=route,
        from_name=((raw.get("from") or {}).get("name") or "").strip(),
        to_name=((raw.get("to") or {}).get("name") or "").strip(),
        minutes=minutes,
    )


async def public_transport_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    *,
    when: Optional[datetime] = None,
) -> Optional[Route]:
    """
    Public-transport itinerary between two points. Returns None when routing is
    unconfigured or OneMap fails, so callers can fall back to a maps link.
    """
    token = await _get_token()
    if not token:
        return None

    cfg = get_onemap_settings()
    when = when or datetime.now(_SGT)
    params = {
        "start": f"{start_lat},{start_lon}",
        "end": f"{end_lat},{end_lon}",
        "routeType": "pt",
        "date": when.strftime("%m-%d-%Y"),
        "time": when.strftime("%H:%M:%S"),
        "mode": "TRANSIT",
        "maxWalkDistance": "1000",
        "numItineraries": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{cfg.base_url}/api/public/routingsvc/route",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("OneMap routing failed: %s", exc)
        return None

    itineraries = (data.get("plan") or {}).get("itineraries") or []
    if not itineraries:
        return None

    it = itineraries[0]
    legs = [_leg_from_raw(raw) for raw in (it.get("legs") or [])]
    try:
        total = max(1, round(float(it.get("duration", 0)) / 60))
        walk = round(float(it.get("walkTime", 0)) / 60)
    except (TypeError, ValueError):
        total, walk = sum(l.minutes for l in legs) or 1, 0

    fare = it.get("fare")
    fare_str = f"${float(fare):.2f}" if fare not in (None, "") else None

    return Route(
        total_minutes=total,
        walk_minutes=walk,
        transfers=int(it.get("transfers") or 0),
        fare=fare_str,
        legs=legs,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_MODE_ICON = {"WALK": "🚶", "BUS": "🚌", "SUBWAY": "🚇", "RAIL": "🚆", "TRAM": "🚊"}


def render_route(route: Route) -> str:
    """Telegram-HTML step list for one person's journey."""
    lines: list[str] = []
    for leg in route.legs:
        icon = _MODE_ICON.get(leg.mode, "➡️")
        if leg.mode == "WALK":
            dest = leg.to_name or "the next stop"
            lines.append(f"{icon} Walk {leg.minutes} min to {dest}")
        else:
            service = f" <b>{leg.line}</b>" if leg.line else ""
            lines.append(
                f"{icon} Take{service} from {leg.from_name or '?'} "
                f"→ {leg.to_name or '?'} ({leg.minutes} min)"
            )
    header = f"⏱ <b>{route.total_minutes} min</b>"
    if route.transfers:
        header += f" · {route.transfers} transfer{'s' if route.transfers > 1 else ''}"
    if route.fare:
        header += f" · {route.fare}"
    if route.walk_minutes:
        header += f" · {route.walk_minutes} min walking"
    return header + "\n" + "\n".join(lines)
