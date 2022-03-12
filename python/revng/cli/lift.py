from tempfile import NamedTemporaryFile
from .revng import run_revng_command


def register_lift(subparsers):
    parser = subparsers.add_parser("lift", description="revng lift wrapper")
    parser.add_argument("input", type=str, nargs=1, help="Input binary to be lifted")
    parser.add_argument("output", type=str, nargs=1, help="Output module path")
    parser.add_argument("--record-asm", action="store_true")
    parser.add_argument("--record-ptc", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--base", type=str)
    parser.add_argument("--entry", type=str)
    parser.add_argument("--debug-info", type=str)
    parser.add_argument("--import-debug-info", type=str, action="append", default=[])


def run_lift(args, post_dash_dash_args, search_prefixes, command_prefix):
    # Run revng model import args.input
    arg_or_empty = (
        lambda args, name: [f"--{name}={args.__dict__[name]}"] if args.__dict__[name] else []
    )

    with NamedTemporaryFile(suffix=".yml") as model, NamedTemporaryFile(
        suffix=".ll"
    ) as model_in_module:
        run_revng_command(
            [
                "revng",
                "model",
                "import",
                "binary",
                args.input[0],
                "-o",
                model.name,
            ]
            + [f"--import-debug-info={value}" for value in args.import_debug_info]
            + arg_or_empty(args, "base")
            + arg_or_empty(args, "entry"),
            search_prefixes,
            command_prefix,
        )

        run_revng_command(
            [
                "revng",
                "model",
                "inject",
                model.name,
                "/dev/null",
                "-o",
                model_in_module.name,
            ],
            search_prefixes,
            command_prefix,
        )

        run_revng_command(
            [
                "revng",
                "opt",
                model_in_module.name,
                "-o",
                args.output[0],
                "-load-binary",
                f"-binary-path={args.input[0]}",
                "-lift",
            ]
            + arg_or_empty(args, "external")
            + arg_or_empty(args, "record_asm")
            + arg_or_empty(args, "record_ptc"),
            search_prefixes,
            command_prefix,
        )

    # TODO: annotate IR
