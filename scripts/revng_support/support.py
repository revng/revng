import sys
import os
import signal
import subprocess
import shlex

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which

def shlex_join(split_command):
    return ' '.join(shlex.quote(arg) for arg in split_command)


def log_error(msg):
    sys.stderr.write(msg + "\n")


def relative(path):
    return os.path.relpath(path, ".")

def wrap(args, command_prefix):
    return command_prefix + args

log_commands = None

def run(command, command_prefix, override={}):
    if is_executable(command[0]):
        command = wrap(command, command_prefix)

    global log_commands
    if log_commands:
        cwd = os.getcwd().rstrip("/")
        if command[0].startswith(cwd):
            program_path = "." + command[0][len(cwd):]
        else:
            program_path = command[0]
        sys.stderr.write("{}\n\n".format(
            " \\\n  ".join([program_path] + command[1:])))

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    environment = dict(os.environ)
    environment.update(override)
    p = subprocess.Popen(command,
                         preexec_fn=lambda: signal.signal(signal.SIGINT,
                                                          signal.SIG_DFL),
                         env=environment)
    if p.wait() != 0:
        log_error("The following command exited with {}:\n{}".format(
            p.returncode, shlex_join(command)))
        sys.exit(p.returncode)


def is_executable(path):
    with open(path, "rb") as program:
        return program.read(4) == b"\x7fELF"


def is_dynamic(path):
    with open(path, "rb") as input_file:
        return len([segment
                    for segment
                    in ELFFile(input_file).iter_segments()
                    if segment.header.p_type == "PT_DYNAMIC"]) != 0


def get_command(command, search_path):
    path = which(command, path=search_path)
    if not path:
        log_error("""Couldn't find "{}" in "{}".""".format(
            command, search_path))
        assert False
    return os.path.abspath(path)

