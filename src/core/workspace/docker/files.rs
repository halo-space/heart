use super::Backend;
use crate::{Cancellation, Error, core::operation::check};

// Paths/data are always positional arguments/stdin, never shell interpolation.
// All tools use this container cwd; find does not follow symlinks. Commands may
// mutate the filesystem, so the same trusted-root caveat as Local applies.
const SCRIPT: &str = r#"
fail() { printf '%s' "$1" >&2; exit "$2"; }
action=$1
path=$2
rest=$path
prefix=.
while :; do
    part=${rest%%/*}
    prefix=$prefix/$part
    [ ! -L "$prefix" ] || fail 'workspace symlinks are not allowed' 91
    case "$rest" in */*) rest=${rest#*/} ;; *) break ;; esac
done
target=./$path
case "$action" in
read)
    [ -e "$target" ] || fail 'workspace file not found' 92
    [ -f "$target" ] || fail 'workspace path is not a regular file' 90
    cat "$target" || fail 'workspace read failed' 93
    ;;
write)
    [ ! -d "$target" ] || fail 'workspace path is a directory' 90
    [ ! -e "$target" ] || [ -f "$target" ] || fail 'workspace path is not a regular file' 90
    parent=${target%/*}
    mkdir -p "$parent" || fail 'workspace parent creation failed' 93
    temp=$parent/.heart-write-$3
    (umask 077; set -C; : > "$temp") || fail 'workspace temporary creation failed' 93
    trap 'rm -f "$temp"' EXIT HUP INT TERM
    cat > "$temp" || fail 'workspace write failed' 93
    mv -f "$temp" "$target" || fail 'workspace write commit failed' 93
    ;;
delete)
    [ -e "$target" ] || fail 'workspace file not found' 92
    [ -f "$target" ] || fail 'workspace path is not a regular file' 90
    rm "$target" || fail 'workspace deletion failed' 93
    ;;
list)
    [ -e "$target" ] || fail 'workspace directory not found' 92
    [ -d "$target" ] || fail 'workspace path is not a directory' 90
    if [ "$3" = true ]; then
        find "$target" -type f -print0 || fail 'workspace listing failed' 93
    else
        find "$target" -mindepth 1 -maxdepth 1 \( -type f -o -type d \) -print0 || fail 'workspace listing failed' 93
    fi
    ;;
*) fail 'unsupported workspace file operation' 90 ;;
esac
"#;

impl Backend {
    async fn file(
        &self,
        action: &str,
        path: &str,
        extra: &str,
        data: Option<&[u8]>,
        cancellation: &Cancellation,
    ) -> Result<Vec<u8>, Error> {
        check(cancellation)?;
        if action != "list" || path != "." {
            super::super::validate_path(path)?;
        }
        let args = [
            "/bin/sh".into(),
            "-c".into(),
            SCRIPT.into(),
            "heart-workspace".into(),
            action.into(),
            path.into(),
            extra.into(),
        ];
        let output = self.execute(&args, data, cancellation).await?;
        if output.code != 0 {
            let code = match output.code {
                90 => "INVALID_ARGUMENTS",
                91 => "PERMISSION_DENIED",
                92 => "NOT_FOUND",
                93 => "IO_ERROR",
                _ => "EXECUTION_FAILED",
            };
            return Err(Error::new(code, "Docker workspace file operation failed"));
        }
        Ok(output.stdout)
    }

    pub(super) async fn read(
        &self,
        path: &str,
        cancellation: &Cancellation,
    ) -> Result<Vec<u8>, Error> {
        self.file("read", path, "", None, cancellation).await
    }

    pub(super) async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let token = format!("{:032x}", rand::random::<u128>());
        self.file("write", path, &token, Some(&data), cancellation)
            .await?;
        Ok(())
    }

    pub(super) async fn delete(
        &self,
        path: &str,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.file("delete", path, "", None, cancellation).await?;
        Ok(())
    }

    pub(super) async fn list(
        &self,
        path: &str,
        recursive: bool,
        cancellation: &Cancellation,
    ) -> Result<Vec<String>, Error> {
        let bytes = self
            .file(
                "list",
                path,
                if recursive { "true" } else { "false" },
                None,
                cancellation,
            )
            .await?;
        if !bytes.is_empty() && bytes.last() != Some(&0) {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "unterminated workspace listing",
            ));
        }
        let mut paths = Vec::new();
        for entry in bytes
            .split(|byte| *byte == 0)
            .filter(|entry| !entry.is_empty())
        {
            let entry = std::str::from_utf8(entry)
                .map_err(|_| Error::new("INVALID_RESPONSE", "workspace path is not UTF-8"))?;
            let entry = entry
                .strip_prefix("././")
                .or_else(|| entry.strip_prefix("./"))
                .ok_or_else(|| {
                    Error::new("INVALID_RESPONSE", "invalid workspace-relative listing")
                })?;
            super::super::validate_path(entry).map_err(|_| {
                Error::new("INVALID_RESPONSE", "invalid workspace-relative listing")
            })?;
            paths.push(entry.to_owned());
        }
        paths.sort();
        Ok(paths)
    }
}
