#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click
import yaml

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, cli_logger
from revng.internal.cli.common import pass_context
from revng.internal.cli.support import file_wrapper, keep_temporaries_option, temporary_file_gen


@click.command(
    cls=WrappableCommand,
    name="hard-purge",
    help="Purge all the functions from original model that does not exist in "
    "the reference model.",
)
@click.argument("reference_model_path", metavar="REFERENCE_MODEL")
@click.argument("original_model_path", metavar="[ORIGINAL_MODEL]", required=False)
@click.option(
    "-o",
    "purged_model_path",
    default="/dev/stdout",
    help="The pruned model in form of YAML.",
)
@keep_temporaries_option
@pass_context
def hard_purge(
    ctx: ClickContext,
    reference_model_path: str,
    original_model_path: str | None,
    purged_model_path: str,
    keep_temporaries: bool,
) -> int:
    functions_to_preserve = set()

    # Collect functions to be preserved.
    with open(reference_model_path, "rb") as reference_model_file:
        cli_logger.debug_log("Loading the reference model...")
        reference_model = yaml.load(reference_model_file, Loader=yaml.SafeLoader)

        if "Functions" in reference_model:
            for function in reference_model["Functions"]:
                function_name = function["Name"]
                cli_logger.debug_log(" Function to be preserved: " + function_name)
                functions_to_preserve.add(function_name)

        if "ImportedDynamicFunctions" in reference_model:
            for dynamic_function in reference_model["ImportedDynamicFunctions"]:
                function_name = dynamic_function["Name"]
                cli_logger.debug_log(" Dynamic function to be preserved: " + function_name)
                functions_to_preserve.add(function_name)

    # Remove the functions.
    cli_logger.debug_log("Removing functions from original mode...")
    patched_model = {}
    with file_wrapper(original_model_path, "r") as patched_file:
        patched_model = yaml.load(patched_file, Loader=yaml.SafeLoader)

        # Delete functions.
        patched_model["Functions"] = [
            f for f in patched_model["Functions"] if f.get("Name", "") in functions_to_preserve
        ]

        # Delete dynamic functions.
        patched_model["ImportedDynamicFunctions"] = [
            f
            for f in patched_model["ImportedDynamicFunctions"]
            if f["Name"] in functions_to_preserve
        ]

    temporary_file = temporary_file_gen("revng-hard-purge-", keep_temporaries)
    with temporary_file(suffix=".yml") as model_file:
        model_file.write("---\n")
        yaml.dump(patched_model, stream=model_file)
        model_file.write("...\n")
        model_file.flush()

        # Optimize the model by purging all unreachable types from any Function.
        return ctx.obj.try_run(
            [
                "revng",
                "model",
                "opt",
                "-purge-unreachable-types",
                model_file.name,
                "-o",
                purged_model_path,
            ]
        )


def setup(registry: CommandRegistry):
    registry.register(("model",), hard_purge)
