#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from shutil import copyfileobj

import click
import yaml

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, cli_logger
from revng.internal.cli.common import pass_context
from revng.internal.cli.support import keep_temporaries_option, temporary_file_gen


@click.command(cls=WrappableCommand, name="override-by-name", help="Override parts of the model")
@click.argument("input_model_path", metavar="INPUT_MODEL")
@click.argument("override_model_path", metavar="OVERRIDE_MODEL")
@click.option("-o", "output", metavar="OUTPUT", default="/dev/stdout", help="Output path")
@keep_temporaries_option
@pass_context
def override_by_name(
    ctx: ClickContext,
    input_model_path: str,
    override_model_path: str,
    output: str,
    keep_temporaries: bool,
) -> int:
    temporary_file = temporary_file_gen("revng-override-by-name-", keep_temporaries)
    with open(input_model_path, "rb") as input_file, temporary_file(
        mode="wb+"
    ) as saved_file, temporary_file(suffix=".yml") as model_file, open(
        override_model_path
    ) as override_file, temporary_file(
        suffix=".yml"
    ) as patched_file, temporary_file(
        suffix=".yml.diff"
    ) as patch_file, temporary_file(
        suffix=".yml"
    ) as patched_model_file:

        # Copy so we can work with stdin too
        copyfileobj(input_file, saved_file)
        saved_file.flush()

        # Check if it's YAML
        saved_file.seek(0)
        input_is_yaml = saved_file.read(3) == b"---"

        # Extract model, if necessary
        cmd_base = ("revng", "model")
        result = ctx.obj.try_run([*cmd_base, "opt", saved_file.name, "-o", model_file.name])
        if result != 0:
            return result

        cli_logger.debug_log("Loading the base model")
        base_model = yaml.load(model_file, Loader=yaml.SafeLoader)
        cli_logger.debug_log("Loading the override model")
        override_model = yaml.load(override_file, Loader=yaml.SafeLoader)

        cli_logger.debug_log("Importing entry address and name")
        for function_to_override in override_model["Functions"]:
            function_name = function_to_override["Name"]

            if not function_name:
                cli_logger.debug_log("A function is missing a name Name")
                return 1

            for base_function in base_model["Functions"]:
                if "Name" not in base_function:
                    continue
                if base_function["Name"] == function_name:
                    function_to_override["Entry"] = base_function["Entry"]
                    function_to_override["Name"] = base_function["Name"]

        cli_logger.debug_log("Saving patched override file")
        patched_file.write("---\n")
        yaml.dump(override_model, stream=patched_file)
        patched_file.write("...\n")
        patched_file.flush()

        cli_logger.debug_log("Compute diff between patched override file and base model")
        result = ctx.obj.try_run(
            [*cmd_base, "diff", model_file.name, patched_file.name, "-o", patch_file.name]
        )

        if result not in [0, 1]:
            return result

        cli_logger.debug_log("Loading the patch file")
        with open(patch_file.name) as loaded_patch_file:
            patch = yaml.load(stream=loaded_patch_file, Loader=yaml.SafeLoader)

        cli_logger.debug_log("Removing all removals from the patch file")
        patch["Changes"] = [
            change
            for change in patch["Changes"]
            if ("Add" in change and change["Add"] not in ["", "Invalid", ":Invalid"])
        ]

        cli_logger.debug_log("Saving the patched patch file")
        with open(patch_file.name, "wt") as saved_patch_file:
            yaml.dump(patch, stream=saved_patch_file)

        cli_logger.debug_log("Applying the patched patch to the base model")

        if input_is_yaml:
            patched_model_path = output
        else:
            patched_model_path = patched_model_file.name

        result = ctx.obj.try_run(
            [*cmd_base, "apply", model_file.name, patch_file.name, "-o", patched_model_path]
        )

        if result != 0:
            return result

        if not input_is_yaml:
            result = ctx.obj.try_run(
                [*cmd_base, "inject", patched_model_path, saved_file.name, "-o", output]
            )

        return result


def setup(registry: CommandRegistry):
    registry.register(("model",), override_by_name)
