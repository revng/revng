# Build and publish CI

`build-and-push.sh` builds `revng` and `test/revng`, then publishes their
closures to the configured binary caches.

Required environment variables:

- `REVNG_CACHE_SSH_TARGET`: SSH destination in `user@host` form.
- `REVNG_CACHE_PUBLIC_SIGNING_KEY`: Nix secret signing key for the public cache.
- `REVNG_CACHE_PRIVATE_SIGNING_KEY`: Nix secret signing key for the private cache.
- `REVNG_CACHE_SSH_PRIVATE_KEY`: SSH private key for the cache account.
- `REVNG_CACHE_SSH_KNOWN_HOSTS`: trusted `known_hosts` entry for `rev.ng`.
- `REVNG_NIX_CACHE_TOKEN`: enables reads from the private cache.
