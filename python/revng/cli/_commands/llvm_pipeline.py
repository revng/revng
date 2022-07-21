#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from revng.cli.commands_registry import Command, Options
from revng.cli.revng import run_revng_command
from revng.cli.support import log_error


class IRPipelineCommand(Command):
    def __init__(self):
        super().__init__(("llvm", "pipeline"), "revng-pipeline working on LLVM IR")

    def register_arguments(self, parser):
        parser.add_argument("input", type=str, default="/dev/stdin")
        parser.add_argument("output", type=str, default="/dev/stdout")
        parser.add_argument("from", type=str)
        parser.add_argument("to", type=str)
        parser.add_argument("target", type=str)
        parser.add_argument("--analysis", type=str, default="")

    def run(self, options: Options):
        if options.remaining_args:
            log_error("Unknown arguments passed in")
            return 1

        args = options.parsed_args

        with NamedTemporaryFile(
            suffix=".yml", delete=not options.keep_temporaries
        ) as model, NamedTemporaryFile(
            suffix=".bc", delete=not options.keep_temporaries
        ) as module, open(  # noqa: SIM115
            args.input, "rb"
        ) if args.input != "-" else sys.stdin.buffer as input_file, NamedTemporaryFile(
            suffix=".bc", delete=not options.keep_temporaries, mode="wb"
        ) as saved_input_file:
            # Copy so we can work with stdin too
            copyfileobj(input_file, saved_input_file)
            saved_input_file.flush()

            # Extract model
            run_revng_command(["model", "dump", saved_input_file.name, "-o", model.name], options)

            # Run revng-pipeline
            target = f"{args.to}:module.ll:{args.target}"
            if args.analysis:
                target = f"{args.to}:{args.analysis}:module.ll:{args.target}"

            run_revng_command(
                [
                    "pipeline",
                    "-m",
                    model.name,
                    "-i",
                    f"""{args.__dict__["from"]}:module.ll:{saved_input_file.name}""",
                    "-o",
                    f"{args.to}:module.ll:{module.name}",
                    "--save-model",
                    model.name,
                    target,
                ],
                options,
            )

            # Re-inject the model
            run_revng_command(
                ["model", "inject", model.name, module.name, "-o", args.output], options
            )
        return 0
