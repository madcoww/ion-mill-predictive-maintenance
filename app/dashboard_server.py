"""Serve the AegisOps dashboard and its cutaway preview."""
import os
from pathlib import Path
from secrets import compare_digest
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pwdlib import PasswordHash
from starlette.middleware.sessions import SessionMiddleware


DEMO_DIR = Path(__file__).resolve().parent / "ion-mill-dashboard"
LOGIN_ID = os.environ["WAFERS_LOGIN_ID"]
LOGIN_PASSWORD_HASH = os.environ["WAFERS_LOGIN_PASSWORD_HASH"]
password_hasher = PasswordHash.recommended()

app = FastAPI(title="AegisOps Dashboard")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["WAFERS_SESSION_SECRET"],
    session_cookie="wafers_session",
    max_age=3600,
)
app.mount("/assets", StaticFiles(directory=DEMO_DIR / "assets"), name="assets")


@app.get("/")
@app.get("/login")
def login_page(request: Request) -> Response:
    headers = {"Cache-Control": "no-store"}

    if request.session.get("user_id") == LOGIN_ID:
        return RedirectResponse("/dashboard", status_code=303, headers=headers)

    return FileResponse(
        DEMO_DIR / "wafers_sign_in.html",
        media_type="text/html",
        headers=headers,
    )

@app.post("/login")
def sign_in(
    request: Request,
    employee_id: Annotated[str, Form(alias="employee-id")],
    password: Annotated[str, Form()],
) -> RedirectResponse:
    id_ok = compare_digest(employee_id.strip().encode(), LOGIN_ID.encode())
    password_ok = password_hasher.verify(password, LOGIN_PASSWORD_HASH)

    if not (id_ok and password_ok):
        return RedirectResponse("/login?error=1", status_code=303)

    request.session.clear()
    request.session["user_id"] = LOGIN_ID
    return RedirectResponse("/dashboard", status_code=303)


def private_page(request: Request, filename: str) -> Response:
    if request.session.get("user_id") != LOGIN_ID:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(
        DEMO_DIR / filename,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/dashboard")
@app.get("/aegisops_theme.html", include_in_schema=False)
def dashboard(request: Request) -> Response:
    return private_page(request, "aegisops_theme.html")


@app.get("/ion-mill.html")
def cutaway(request: Request) -> Response:
    return private_page(request, "ion-mill.html")


@app.post("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/login", status_code=303)