#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from starlette.templating import Jinja2Templates

module_dir = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=module_dir / "templates")


async def demo_page(request):
    return templates.TemplateResponse("index.html", {"request": request})
