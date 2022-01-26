import sys
import os
import glob
import re
import signal
import subprocess
import shlex
from itertools import chain
from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which

def shlex_join(split_command): return ' '.join(shlex.quote(arg) for arg in split_command)


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


def interleave(base, repeat):
    return list(sum(zip([repeat] * len(base), base), ()))

# Use in case different version of pyelftools might give str or bytes

def to_string(obj):
    if type(obj) is str:
        return obj
    elif type(obj) is bytes:
        return obj.decode("utf-8")


def get_elf_needed(path):
    with open(path, "rb") as elf_file:
        segments = [segment
                    for segment
                    in ELFFile(elf_file).iter_segments()
                    if type(segment) is DynamicSegment]

        if len(segments) == 1:
            needed = [to_string(tag.needed)
                      for tag
                      in segments[0].iter_tags()
                      if tag.entry.d_tag == "DT_NEEDED"]

            runpath = [tag.runpath
                       for tag
                       in segments[0].iter_tags()
                       if tag.entry.d_tag == "DT_RUNPATH"]

            assert len(runpath) < 2

            if not runpath:
                return needed
            else:
                runpaths = [runpath.replace("$ORIGIN",
                                            os.path.dirname(path))
                            for runpath
                            in runpath[0].split(":")]
                absolute_needed = []
                for lib in needed:
                    found = False
                    for runpath in runpaths:
                        full_path = os.path.join(runpath, lib)
                        if os.path.isfile(full_path):
                            absolute_needed.append(full_path)
                            found = True
                            break

                    if not found:
                        absolute_needed.append(lib)

                return absolute_needed

        else:
            return []

def discover_all_used_libraries(search_prefixes):
    all = {}
    for prefix in search_prefixes:
        analyses_path = os.path.join(prefix, "lib", "revng", "analyses")
        if not os.path.isdir(analyses_path):
            continue

        # Enumerate all the libraries containing analyses
        for library in glob.glob(os.path.join(analyses_path, "*.so")):
            basename = os.path.basename(library)
            if basename not in all:
                all[basename] = library

    # Identify all the libraries that are dependencies of other libraries, i.e.,
    # non-roots in the dependencies tree. Note that circular dependencies are
    # not allowed.
    dependencies = set(chain.from_iterable([get_elf_needed(path)
                                        for path
                                        in all.values()]))
    return (all, dependencies)

def discover_all_root_libraries(all, dependencies):
    # Path to libraries representing the roots in the dependency tree
    provided_so_names = set(os.path.basename(lib) for lib in dependencies)
    return [relative(path)
             for basename, path
             in all.items()
             if basename not in provided_so_names]

def build_prefix(dependencies):
    libasan = [name
               for name
               in dependencies
               if ("libasan." in name
                   or "libclang_rt.asan" in name)]

    if len(libasan) != 1:
        return []

    libasan_path = relative(libasan[0])
    original_asan_options = os.environ.get("ASAN_OPTIONS", "")
    if original_asan_options:
        asan_options = dict([option.split("=")
                             for option
                             in original_asan_options.split(":")])
    else:
        asan_options = dict()
    asan_options["abort_on_error"] = "1"
    new_asan_options = ":".join(["=".join(option)
                                 for option
                                 in asan_options.items()])

    # Use `sh` instead of `env` since `env` sometimes is not a real executable
    # but a shebang script spawning /usr/bin/coreutils, which makes gdb unhappy
    return [get_command("sh", search_path),
              "-c",
              'LD_PRELOAD={} ASAN_OPTIONS={} '
              'exec "$0" "$@"'.format(libasan_path, new_asan_options)]

def build_command_with_loads(command, args, search_path):
    (all, dependencies) = discover_all_used_libraries(search_path)
    roots = discover_all_root_libraries(all, dependencies)
    prefix = build_prefix(dependencies)

    return (prefix + [relative(get_command(command, search_path))]
            + interleave(roots, "-load")
            + args)

