#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import yaml

from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.revng import run_revng_command
from revng.cli.support import log_error


class ModelOverrideByName(Command):
    def __init__(self):
        super().__init__(("model", "override-by-name"), "Override parts of the model")

    def register_arguments(self, parser):
        parser.add_argument(
            "input_model_path", metavar="INPUT_MODEL", default="", help="Input model (can be IR)"
        )
        parser.add_argument(
            "override_model_path", metavar="OVERRIDE_MODEL", default="", help="Override model"
        )
        parser.add_argument(
            "-o",
            metavar="OUTPUT",
            dest="output",
            type=str,
            default="/dev/stdout",
            help="Output path",
        )
        parser.add_argument(
            "--remove-others",
            action="store_true",
            help="Remove functions that are not in the override model",
        )

    def log(self, message):
        if self.verbose:
            sys.stderr.write(message + "\n")

    def run(self, options: Options):
        if options.remaining_args:
            log_error("Unknown arguments passed in")
            return 1

        args = options.parsed_args

        self.verbose = args.verbose

        def temporary_file(suffix="", mode="w+"):
            return NamedTemporaryFile(
                prefix="revng-override-by-name-",
                suffix=suffix,
                mode=mode,
                delete=not options.keep_temporaries,
            )

        with open(args.input_model_path, "rb") as input_file, temporary_file(
            mode="wb+"
        ) as saved_file, temporary_file(suffix=".yml") as model_file, open(
            args.override_model_path
        ) as override_file, temporary_file(
            suffix=".yml"
        ) as patched_file, temporary_file(
            suffix=".yml.diff"
        ) as patch_file, temporary_file(
            suffix=".yml"
        ) as patch_for_removal:

            # Copy so we can work with stdin too
            copyfileobj(input_file, saved_file)
            saved_file.flush()

            # Check if it's YAML
            saved_file.seek(0)
            input_is_yaml = saved_file.read(4) != b"BC\xc0\xde"

            # Extract model, if necessary
            result = run_revng_command(
                ["model", "opt", "-Y", saved_file.name, "-o", model_file.name], options
            )
            if result != 0:
                return result

            self.log("Loading the base model")
            base_model = yaml.load(model_file, Loader=yaml.SafeLoader)
            self.log("Loading the override model")
            override_model = yaml.load(override_file, Loader=yaml.SafeLoader)

            self.log("Importing Entry and OriginalName")
            for function_to_override in override_model["Functions"]:
                function_name = function_to_override["OriginalName"]

                if not function_name:
                    self.log("A function is missing OriginalName")
                    return 1

                for base_function in base_model["Functions"]:
                    if base_function["OriginalName"] == function_name:
                        function_to_override["Entry"] = base_function["Entry"]

            self.log("Saving patched override file")
            patched_file.write("---\n")
            yaml.dump(override_model, stream=patched_file)
            patched_file.write("...\n")
            patched_file.flush()

            self.log("Compute diff between patched override file and base model")
            result = run_revng_command(
                ["model", "diff", model_file.name, patched_file.name, "-o", patch_file.name],
                options,
            )

            if result not in [0, 1]:
                return result

            self.log("Loading the patch file")
            with open(patch_file.name) as loaded_patch_file:
                patch = yaml.load(stream=loaded_patch_file, Loader=yaml.SafeLoader)

            self.log("Removing all removals from the patch file")
            patch["Changes"] = [
                change
                for change in patch["Changes"]
                if ("Add" in change and change["Add"] not in ["", "Invalid", ":Invalid"])
            ]

            self.log("Saving the patched patch file")
            with open(patch_file.name, "wt") as saved_patch_file:
                yaml.dump(patch, stream=saved_patch_file)

            self.log("Applying the patched patch to the base model")

            result = run_revng_command(
                ["model", "apply", model_file.name, patch_file.name, "-o", patch_for_removal.name],
                options,
            )

            if args.remove_others:
                self.log(
                    "Removing functions and dynamic functions not appearing in the override file"
                )

                with open(patch_for_removal.name) as patched_file:
                    base_patch = yaml.load(stream=patched_file, Loader=yaml.SafeLoader)

                override_model_function_names = [
                    function["OriginalName"] for function in override_model["Functions"]
                ]

                override_model_dynamicfunctions_names = [
                    function["OriginalName"]
                    for function in override_model["ImportedDynamicFunctions"]
                ]

                base_patch["Functions"] = [
                    item
                    for item in base_patch["Functions"]
                    if item["OriginalName"] in override_model_function_names
                ]

                base_patch["ImportedDynamicFunctions"] = [
                    item
                    for item in base_patch["ImportedDynamicFunctions"]
                    if item["OriginalName"] in override_model_dynamicfunctions_names
                ]

                with open(patch_for_removal.name, "w") as saved_patch_file:
                    saved_patch_file.write("---\n")
                    yaml.dump(base_patch, stream=saved_patch_file)
                    saved_patch_file.write("...\n")

            if result != 0:
                return result

            if not input_is_yaml:
                result = run_revng_command(
                    ["model", "inject", patch_for_removal.name, saved_file.name, "-o", args.output],
                    options,
                )
            else:
                with open(patch_for_removal.name, "rb") as input_f, open(
                    args.output, "wb"
                ) as output_f:
                    copyfileobj(input_f, output_f)

            return result


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ModelOverrideByName())
