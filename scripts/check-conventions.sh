#!/bin/bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORMAT="0"

set -e

while [[ $# > 0 ]]; do
    key="$1"
    case $key in
        --format)
            FORMAT="1"
            shift
            ;;
        --force-format)
            FORMAT="2"
            shift # past argument
            ;;
        --*)
            echo "Unexpected option $key" > /dev/stderr
            echo > /dev/stderr
            echo "Usage: $0 [--format] [--force-format] [FILE...]" > /dev/stderr
            exit 1
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    FILES="$(git ls-files | grep -E '(\.cpp|\.c|\.h)$')"
else
    FILES="$@"
fi
GREP="git grep -n --color=always"

if test "$FORMAT" -gt 0; then
    if test "$FORMAT" -eq 1 && ! git diff --exit-code > /dev/null; then
        echo "Can't run clang-format -i: there are unstaged changes!" > /dev/stderr
        echo 'Run `git reset --hard` or use --force-format to run it anyway.' > /dev/stderr
        exit 1
    fi
    clang-format -style=file -i $FILES
fi

(
    # Check for lines longer than 80 columns
    $GREP -E '^.{81,}$' $FILES | cat

    # Things should never match
    for REGEXP in '\(--> 0\)' ';;' '^\s*->.*;$' 'Twine [^&]'; do
        $GREP "$REGEXP" $FILES | cat
    done

    # Things should never match (except in support.c)
    FILTERED_FILES="$(echo $FILES | sed 's|\bruntime/support\.c\b||g; s|\blib/Support/Assert\.cpp\b||g;')"
    for REGEXP in '\babort(' '\bassert(' 'assert(false' 'llvm_unreachable'; do
        $GREP "$REGEXP" $FILTERED_FILES | cat
    done

    # Things should never be at the end of a line
    for REGEXP in '::' '<' 'RegisterPass.*>' '} else' '\bopt\b.*>'; do
        $GREP "$REGEXP\$" $FILES | cat
    done

    # Parenthesis at the end of line (except for raw strings)
    $GREP "(\$" $FILES | grep -v 'R"LLVM.*(' | cat

    # Things should never be at the beginning of a line
    for REGEXP in '\.[^\.]' '\*>' '/[^/\*]' ':[^:\(]*)' '==' '\!=' '<[^<]' '>' '>=' '<=' '//\s*WIP' '#if\s*[01]'; do
        $GREP "^\s*$REGEXP" $FILES | cat
    done

    # Check there are no static functions in header files
    for FILE in $FILES; do
        if [[ $FILE == *h ]]; then
            $GREP -H '^static[^=]*$' "$FILE" | cat
            $GREP -HE '#ifndef\s*_.*_H\s*$' "$FILE" | cat
        fi
    done
) | sort -u
