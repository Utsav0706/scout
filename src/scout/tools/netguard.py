"""SSRF protection for fetch_url.

The agent chooses which URLs to fetch, and a page it reads could tell it to fetch
http://169.254.169.254/ (the AWS metadata endpoint that hands out credentials) or
http://localhost:5432. So before every request, and again on every redirect hop:
- only http/https on the default ports (80/443),
- the hostname must resolve, and EVERY address it resolves to must be public.

Known gap: httpx resolves the name again when it connects, so a DNS server that answers
"public" to us and "private" to httpx (DNS rebinding) could slip through. Fully closing
it means pinning the connection to the checked IP; acceptable risk for now, and the
deployed container has no credentials on its metadata endpoint worth stealing.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import anyio

from scout.tools.context import Resolver


class BlockedURLError(ValueError):
    pass


_ALLOWED = {"http": 80, "https": 443}


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _is_public(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%")[0])  # strip IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is really 127.0.0.1
    # is_global is False for private, loopback, link-local (169.254.x), CGNAT,
    # multicast, reserved and unspecified ranges.
    return ip.is_global


async def check_url(url: str, resolver: Resolver) -> None:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED:
        raise BlockedURLError(f"Only http and https URLs are allowed, got '{scheme or 'none'}'")
    if not parts.hostname:
        raise BlockedURLError("URL has no host")
    if parts.username or parts.password:
        raise BlockedURLError("URLs with embedded credentials are not allowed")
    try:
        port = parts.port or _ALLOWED[scheme]
    except ValueError:
        raise BlockedURLError("Invalid port") from None
    if port != _ALLOWED[scheme]:
        raise BlockedURLError(f"Only the default port for {scheme} is allowed")

    try:
        addresses = await resolver(parts.hostname, port)
    except OSError:
        raise BlockedURLError(f"Could not resolve host '{parts.hostname}'") from None
    if not addresses:
        raise BlockedURLError(f"Could not resolve host '{parts.hostname}'")
    if not all(_is_public(a) for a in addresses):
        raise BlockedURLError("URL points to a private or internal address")
