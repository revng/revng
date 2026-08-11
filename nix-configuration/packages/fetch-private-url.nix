{ pkgs }:
# Wrapper around `pkgs.fetchurl` that tags the resulting FOD with
# `useCache = "private"`, so cache/CI infrastructure knows to route
# it to the private binary cache instead of the public one. Use for
# any download of non-redistributable content (MSVC installers,
# Microsoft eval ISOs, Apple SDK tarballs, …).
#
# `pkgs.fetchurl`'s argument set is strictly validated (via
# `extendDrvArgs`), so `useCache` can't be passed inline; we
# attach it with `overrideAttrs` after the fact, which lands the
# attribute on the derivation (and thus in the emitted `.drv`).
args:
(pkgs.fetchurl args).overrideAttrs (_: {
  __structuredAttrs = true;
  useCache = "private";
})
