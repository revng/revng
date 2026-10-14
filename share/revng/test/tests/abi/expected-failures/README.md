# Expected Failure lists

This directory contains lists of known and expected failures in the runtime ABI testing suite.

It is separated into a set of YAML files, each one describing a self-contained set of failures.

Each file is a sequence of "expectations."

Each expectation is:

```yaml
- ABIs: [abi-1, abi-2, ...]
  tests: [test-1, test-2, ...]
  errors: [error-1, error-2, ...]
  can-succeed: false
```

where:
- `ABIs`: list of ABI names the entry applies to (see `model::ABI::getName` for the legal spellings, e.g. `SystemV_x86_64`, `AAPCS64`, `AAPCS_hardfloat`, `SystemZ_s390x`)
- `tests`: list of test names the entry applies to. A test name is the *function name* of a function in the runtime ABI artifact (see `functions.yml` on revng-qa side for the list).
- `errors`: list of accepted failure messages. Note that full-string matching is used - no regexes, no substrings - the error message must much exactly. An empty list accepts *any* message, as long as the test does *not* succeed.
- `can-succeed` (optional, defaults to `false`): if set to `true`, the entry is not *required* to fail. This is intended as a way to blanket-cover a still-WIP area.

An entry covers the cross product of ABIs x tests. Every `(ABI, test)` pair it
covers requires that ONE of `errors` occurs.

Both directions are checked, and both produce a nonzero exit code:

  (a) a failure that no entry accounts for is reported (a real regression);
  (b) an entry whose (ABI, test) pair matched nothing is *also* reported.

Note that each file has a short description at the top, giving some details into
why specific failures are marked as expected.
