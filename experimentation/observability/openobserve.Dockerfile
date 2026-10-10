# OpenObserve reads its root login only from environment variables (no *_FILE support)
# and its image has no shell. This wrapper adds a tiny static shell (busybox) whose
# entrypoint reads the Docker secrets, puts them in the environment, and starts OpenObserve.
# The login never appears in the compose file, the image, or `docker inspect`.
FROM busybox:stable-musl AS busybox

FROM public.ecr.aws/zinclabs/openobserve:latest
COPY --from=busybox /bin/busybox /busybox
COPY openobserve-entrypoint.sh /openobserve-entrypoint.sh
ENTRYPOINT ["/busybox", "sh", "/openobserve-entrypoint.sh"]
CMD ["/openobserve"]
