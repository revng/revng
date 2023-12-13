#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.revng import run_revng_command


class TranslateCommand(Command):
    def __init__(self):
        super().__init__(("translate",), "revng translate wrapper")

    def register_arguments(self, parser):
        parser.add_argument("input", type=str, nargs=1, help="Input binary to be translated")

        parser.add_argument(
            "-o",
            "--output",
            metavar="OUTPUT",
            dest="output",
            type=str,
            nargs="?",
            help="Translated output",
        )

        parser.add_argument(
            "-i", "--isolate", action="store_true", help="Enable function isolation."
        )

        parser.add_argument(
            "--trace", action="store_true", help="Use the tracing version of support.ll."
        )

        parser.add_argument("-O0", action="store_true", help="Do no optimize.")
        parser.add_argument("-O1", action="store_true", help="Use llc -O2.")
        parser.add_argument("-O2", action="store_true", help="Use llc -O2 and opt -O2.")

        parser.add_argument("--base", help="Load address to employ in lifting.")

    def run(self, options: Options):
        args = options.parsed_args
        out_file = args.output if args.output else args.input[0] + ".translated"

        step_name = "Recompile"
        if args.isolate:
            step_name = step_name + "Isolated"

        command = [
            "pipeline",
        ]

        command.append("--analyze=Import/ImportBinary/input/:Binary")
        command.append("--analyze=Import/AddPrimitiveTypes/")
        command.append("--analyze=Lift/DetectABI/module.ll/:Root")

        command = command + [
            f"--produce={step_name}/output/:Translated",
            "-i",
            f"{args.input[0]}:begin/input",
            "-o",
            f"{out_file}:{step_name}/output",
        ]

        if args.O1:
            command.append("--compile-opt-level=1")
        elif args.O2:
            command += ["--compile-opt-level=2", "-f", "O2"]
        else:
            command.append("--compile-opt-level=0")

        if args.base:
            command.append(f"--base={args.base}")

        if args.trace:
            command.append("--link-trace")

        return run_revng_command(command, options)


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(TranslateCommand())
