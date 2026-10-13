#!/busybox sh
# Start OpenObserve with its root login taken from Docker secrets.
# `read` is a shell builtin, so nothing else is needed in this minimal image.
# A secret file may end without a newline, which makes `read` return failure at
# end-of-file even though it filled the variable; `|| [ -n "$VAR" ]` accepts that,
# and still fails (under `set -e`) if the file is empty.
set -eu

read -r ZO_ROOT_USER_EMAIL < /run/secrets/openobserve_user || [ -n "$ZO_ROOT_USER_EMAIL" ]
read -r ZO_ROOT_USER_PASSWORD < /run/secrets/openobserve_password || [ -n "$ZO_ROOT_USER_PASSWORD" ]
export ZO_ROOT_USER_EMAIL ZO_ROOT_USER_PASSWORD

exec "$@"
