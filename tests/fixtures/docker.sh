#!/bin/sh
# Test-only Docker protocol fixture. Never runs the provided command.
base="$0"
mode=$(cat "$base.mode")
printf '%s\n' "$@" >> "$base.log"
id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
case "$1" in
  create)
    if [ "$mode" = offline ]; then printf '%s\n' 'daemon unavailable' >&2; exit 1; fi
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --label ]; then shift; printf '%s' "${1#*=}" > "$base.label"; fi
      shift
    done
    if [ "$mode" = wait-create ]; then
      printf '%s\n' 'create-ready' >> "$base.log"
      exec sleep 30
    fi
    if [ "$mode" = bad-id ]; then printf '%s\n' 'bad-id'; else printf '%s\n' "$id"; fi
    ;;
  start)
    case "$mode" in
      wait) exec sleep 30 ;;
      failed) printf '%s' 'bad command' >&2; exit 2 ;;
      large) printf '%s' 'a very large result' ;;
      *) printf '%s' 'hello' ;;
    esac
    ;;
  inspect)
    if [ "$mode" = offline ]; then printf '%s\n' 'daemon unavailable' >&2; exit 1; fi
    case "$3" in
      *'.State'*)
        code=0; if [ "$mode" = failed ]; then code=2; fi
        printf '{"Running":false,"ExitCode":%s,"OOMKilled":false,"Error":""}\n' "$code"
        ;;
      *)
        token=$(cat "$base.label")
        if [ "$mode" = foreign ]; then token=foreign; fi
        printf '["%s","%s"]\n' "$id" "$token"
        ;;
    esac
    ;;
  rm) printf '%s\n' "$id" ;;
  *) printf '%s\n' 'unexpected Docker operation' >&2; exit 2 ;;
esac
