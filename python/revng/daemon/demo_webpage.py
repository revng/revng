#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.templating import Jinja2Templates

from revng.api import Manager

module_dir = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=module_dir / "templates")


def generate_demo_page(manager: Manager, debug: bool):
    if debug:

        async def dev_demo_page(request: Request):
            return templates.TemplateResponse(
                "index.html", {"request": request, "manager": manager}
            )

        return dev_demo_page
    else:

        async def production_demo_page(request):
            return PlainTextResponse("")

        return production_demo_page
