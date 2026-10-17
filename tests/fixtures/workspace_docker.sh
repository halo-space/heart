#!/bin/sh
# Protocol fixture only: this is NOT a Docker daemon/isolation test. Fixed test
# commands and the actual file scripts run in an owned temporary directory.
set -eu
base=$(dirname "$0")/../state
printf '%s\n' "$1" >> "$base/operations"
printf 'ARG:%s\n' "$@" >> "$base/arguments"
operation=$1
shift
case "$operation" in
create)
    name= token=
    while [ "$#" -gt 0 ]; do
        case "$1" in
        --name) name=$2; shift 2 ;;
        --label) token=${2#*=}; shift 2 ;;
        *) shift ;;
        esac
    done
    [ -n "$name" ] && [ -n "$token" ]
    while ! mkdir "$base/create.lock" 2>/dev/null; do sleep 0.01; done
    count=$(find "$base" -name '*.name' | wc -l | tr -d ' ')
    id=$(printf '%064d' "$((count + 1))")
    printf '%s' "$name" > "$base/$id.name"
    printf '%s' "$token" > "$base/$id.token"
    mkdir "$base/$id.files"
    rmdir "$base/create.lock"
    if [ -f "$base/create_fail" ]; then
        printf 'cannot create ENV_SECRET_MUST_NOT_LEAK' >&2
        exit 1
    fi
    printf '%s\n' "$id"
    ;;
start)
    [ ! -f "$base/start_fail" ] || exit 1
    : > "$base/$1.running"
    printf '%s\n' "$1"
    ;;
exec)
    [ "${1:-}" != --interactive ] || shift
    [ "$1" = --workdir ]
    shift 2
    [ "$1" = -- ]
    shift
    id=$1
    shift
    [ -f "$base/$id.running" ] || exit 1
    cd "$base/$id.files"
    : > "$base/exec_started"
    if [ -f "$base/hang" ]; then exec sleep 20; fi
    while [ -f "$base/wait" ]; do sleep 0.01; done
    if [ -f "$base/transport_fail" ]; then printf 'ENV_SECRET_MUST_NOT_LEAK' >&2; exit 1; fi
    "$@"
    ;;
inspect)
    if [ -f "$base/absent" ]; then printf 'error: no such object' >&2; exit 1; fi
    [ "$1" = --format ]
    format=$2
    target=$3
    if [ "${#target}" -eq 64 ] && [ -f "$base/$target.name" ]; then
        id=$target
    else
        id=
        for file in "$base/"*.name; do
            [ -f "$file" ] || continue
            if [ "$(cat "$file")" = "$target" ]; then id=$(basename "$file" .name); break; fi
        done
    fi
    if [ -z "$id" ]; then printf 'No such object' >&2; exit 1; fi
    if [ "$format" = '{{json .State}}' ]; then
        if [ -f "$base/stopped" ]; then printf '{"Running":false,"OOMKilled":false}';
        elif [ -f "$base/oom" ]; then printf '{"Running":false,"OOMKilled":true}';
        else printf '{"Running":true,"OOMKilled":false}'; fi
    else
        token=$(cat "$base/$id.token")
        [ ! -f "$base/foreign" ] || token=foreign
        printf '["%s","%s"]\n' "$id" "$token"
    fi
    ;;
rm)
    [ "$1" = --force ] && [ "$2" = --volumes ]
    id=$3
    [ "${#id}" -eq 64 ] && [ -f "$base/$id.name" ]
    : > "$base/$id.removed"
    # Preserve files for assertions. The enclosing fixture owns their cleanup.
    rm -f "$base/$id.running"
    ;;
*) printf 'unexpected operation' >&2; exit 2 ;;
esac
