"""LinkedIn authentication, post formatting, and publishing helpers."""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
try:
    from dotenv import load_dotenv, set_key
except ImportError:  # Supports environments with an older dotenv package layout.
    from dotenv.main import load_dotenv, set_key


API_BASE = "https://api.linkedin.com"
AUTH_BASE = "https://www.linkedin.com"
USERINFO_URL = f"{API_BASE}/v2/userinfo"
POSTS_URL = f"{API_BASE}/rest/posts"
TOKEN_URL = f"{AUTH_BASE}/oauth/v2/accessToken"
AUTHORIZE_URL = f"{AUTH_BASE}/oauth/v2/authorization"
SCOPES = "openid profile w_member_social"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class LinkedInSettings:
    """LinkedIn application and token settings."""

    client_id: str
    client_secret: str
    redirect_uri: str
    username: str = ""
    password: str = ""
    access_token: str = ""
    version: str = "202511"


def linkedin_settings(env_file: Path = ENV_FILE) -> LinkedInSettings:
    """Load LinkedIn settings from environment variables and ``.env``."""
    load_dotenv(env_file, override=False)
    required = ("CLIENT_ID", "CLIENT_SECRET", "REDIRECT_URI")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    version = os.getenv("LINKEDIN_VERSION", "202511")
    if len(version) != 6 or not version.isdigit():
        raise RuntimeError("LINKEDIN_VERSION must use YYYYMM format, for example 202511")
    return LinkedInSettings(
        client_id=os.environ["CLIENT_ID"],
        client_secret=os.environ["CLIENT_SECRET"],
        redirect_uri=os.environ["REDIRECT_URI"],
        username=os.getenv("LINKEDIN_USERNAME", "").strip(),
        password=os.getenv("LINKEDIN_PASSWORD", ""),
        access_token=os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip(),
        version=version,
    )


def normalize_hashtags(hashtags: Iterable[str] | str | None) -> list[str]:
    """Return unique hashtags in LinkedIn-ready form, without duplicate ``#``."""
    if hashtags is None:
        return []
    values = [hashtags] if isinstance(hashtags, str) else hashtags
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip()
        if not tag:
            continue
        tag = "#" + tag.lstrip("#").strip()
        if tag == "#":
            continue
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            result.append(tag)
    return result


def compose_post(full_post: str, hashtags: Iterable[str] | str | None = None) -> str:
    """Combine generated post text and hashtags without duplicating tags."""
    post = str(full_post or "").strip()
    if not post:
        raise ValueError("Post text cannot be empty")
    tags = normalize_hashtags(hashtags)
    existing = {match.casefold() for match in re.findall(r"#[\w-]+", post)}
    missing = [tag for tag in tags if tag.casefold() not in existing]
    if missing:
        post = f"{post}\n\n{' '.join(missing)}"
    return post


class LinkedInClient:
    """Publish public text posts through LinkedIn OAuth and the Posts API."""

    def __init__(self, settings: LinkedInSettings, env_file: Path = ENV_FILE) -> None:
        self.settings = settings
        self.env_file = env_file

    def token_is_valid(self, token: str) -> bool:
        if not token:
            return False
        try:
            response = requests.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def _exchange_code(self, code: str) -> str:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.settings.redirect_uri,
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
            },
            timeout=30,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("LinkedIn token response did not contain access_token")
        return token

    def _save_token(self, token: str) -> None:
        set_key(str(self.env_file), "LINKEDIN_ACCESS_TOKEN", token, quote_mode="never")

    def _oauth(self, headless: bool) -> str:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for LinkedIn OAuth; install dependencies from requirements.txt"
            ) from exc

        params = {
            "response_type": "code",
            "client_id": self.settings.client_id,
            "redirect_uri": self.settings.redirect_uri,
            "scope": SCOPES,
        }
        auth_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
        code_holder: dict[str, str] = {}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            redirect = self.settings.redirect_uri

            def capture(route, request):
                if request.url.startswith(redirect) and "code=" in request.url:
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(request.url).query)
                    if query.get("code"):
                        code_holder["code"] = query["code"][0]
                        route.fulfill(
                            status=200,
                            content_type="text/html",
                            body="Login successful; you may close this tab.",
                        )
                        return
                route.continue_()

            if urllib.parse.urlparse(redirect).hostname in {"localhost", "127.0.0.1"}:
                context.route("**/*", capture)
            page.goto(auth_url, wait_until="domcontentloaded", timeout=60_000)
            if self.settings.username and self.settings.password:
                try:
                    page.fill("input#username", self.settings.username)
                    page.click("button[type='submit']")
                    page.wait_for_selector("input#password", timeout=15_000)
                    page.fill("input#password", self.settings.password)
                    page.click("button[type='submit']")
                except PlaywrightTimeoutError:
                    pass
            try:
                page.wait_for_selector("button:has-text('Allow')", timeout=120_000)
                page.click("button:has-text('Allow')")
            except PlaywrightTimeoutError as exc:
                browser.close()
                raise RuntimeError(
                    "Could not complete LinkedIn consent; finish verification in the browser"
                ) from exc
            if "code" not in code_holder:
                try:
                    page.wait_for_url(
                        re.compile(rf"^{re.escape(redirect)}.*code="), timeout=30_000
                    )
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(page.url).query)
                    code_holder["code"] = query["code"][0]
                except PlaywrightTimeoutError:
                    pass
            browser.close()
        if not code_holder.get("code"):
            raise RuntimeError("Could not capture the LinkedIn authorization code")
        token = self._exchange_code(code_holder["code"])
        self._save_token(token)
        return token

    def _token(self, headless: bool) -> str:
        if self.token_is_valid(self.settings.access_token):
            return self.settings.access_token
        token = self._oauth(headless)
        if not self.token_is_valid(token):
            raise RuntimeError("Newly obtained LinkedIn token failed validation")
        return token

    def publish(
        self,
        text: str,
        headless: bool = False,
        hashtags: Iterable[str] | str | None = None,
    ) -> dict[str, str | None]:
        """Publish a public text post, appending any supplied hashtags."""
        content = compose_post(text, hashtags)
        token = self._token(headless)
        userinfo = requests.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        userinfo.raise_for_status()
        author = f"urn:li:person:{userinfo.json()['sub']}"
        response = requests.post(
            POSTS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
                "LinkedIn-Version": self.settings.version,
            },
            json={
                "author": author,
                "commentary": content,
                "visibility": "PUBLIC",
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": [],
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        return {"status": "success", "post_id": response.headers.get("x-restli-id")}
