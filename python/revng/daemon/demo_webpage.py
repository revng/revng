#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.templating import Jinja2Templates

module_dir = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=module_dir / "templates")


def generate_demo_page(workdir: Path, debug: bool):
    if debug:

        async def dev_demo_page(request: Request):
            return templates.TemplateResponse(
                "index.html", {"request": request, "workdir": workdir}
            )

        return dev_demo_page
    else:

        async def production_demo_page(request):
            return PlainTextResponse("")

        return production_demo_page
