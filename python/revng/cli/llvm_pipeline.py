from tempfile import NamedTemporaryFile

from .commands_registry import Command, commands_registry, Options
from .revng import run_revng_command
from .support import log_error


class IRPipelineCommand(Command):
    def __init__(self):
        super().__init__(("llvm", "pipeline"), "revng-pipeline working on LLVM IR")

    def register_arguments(self, parser):
        parser.add_argument("input", type=str, default="/dev/stdin")
        parser.add_argument("output", type=str, default="/dev/stdout")
        parser.add_argument("from", type=str)
        parser.add_argument("to", type=str)
        parser.add_argument("target", type=str)

    def run(self, options: Options):
        if options.remaining_args:
            log_error("Unknown arguments passed in")
            return 1

        args = options.parsed_args

        with NamedTemporaryFile(
            suffix=".yml", delete=not options.keep_temporaries
        ) as model, NamedTemporaryFile(suffix=".bc", delete=not options.keep_temporaries) as module:
            # Extract model
            run_revng_command(["model", "dump", args.input, "-o", model.name], options)

            # Run revng-pipeline
            run_revng_command(
                [
                    "pipeline",
                    "-m",
                    model.name,
                    "-i",
                    f"""{args.__dict__["from"]}:module:{args.input}""",
                    "-o",
                    f"{args.to}:module:{module.name}",
                    "--save-model",
                    model.name,
                    "--step",
                    f"{args.to}",
                    f"module:{args.target}",
                ],
                options,
            )

            # Re-inject the model
            run_revng_command(
                ["model", "inject", model.name, module.name, "-o", args.output], options
            )


commands_registry.register_command(IRPipelineCommand())
