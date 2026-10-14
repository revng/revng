This directory contains lists of known and expected failures in the runtime ABI testing suite.

Each file is an array of "expectations" where each expectation is:

```yaml
- ABIs: [abi-1, abi-2, ...]
  tests: [test-1, test-2, ...]
  errors: [error-1, error-2, ...]
  can-succeed: false
```

where:
- `ABIs`: list of ABI names the entry applies to (see `model::ABI::getName` for the legal spellings, e.g. `SystemV_x86_64`, `AAPCS64`, `AAPCS_hardfloat`, `SystemZ_s390x`)
- `tests`: list of test names the entry applies to. A test name is the *function name* of a function in the runtime ABI artifact (see `functions.yml` on revng-qa side for the list).
- `errors`: list of accepted failure messages. Matching is exact, full-string equality - no regex, no substring. An empty list accepts *any* message, as long as the failure is there.
- `can-succeed` (optional, defaults to `false`): if set to `true`, the entry is not *required* to fail. This is intended as a way to blanket-cover a still-WIP area.

An entry covers the cross product of ABIs x tests. Every (ABI, test) pair it
covers requires that ONE of `errors` to occur.

Both directions are checked, and both produce a nonzero exit code:

  (a) a failure that no entry accounts for is reported (a real regression);
  (b) an entry whose (ABI, test) pair matched nothing is *also* reported.

Here are short descriptions of what each file is about.

##### `mips-specific.yml`

These are failures specific to MIPS. They are expected to go away once we fix the MIPS testing suite.

Namely, return values and stack are not captured correctly.

Also, there are some additional issues with big-endian mips that do not affect little-endian mipsel.

Note that since these issues are visible in the output of the test binaries, they are not necessarily means there are problems with how revng handles these ABIs. BUT they do me that we cannot test these ABIs. So they are definitely hiding bugs!

##### `arm-specific.yml` and `apple-specific.yml`

These are failures specific to ARM, and they are relatively new (uncovered in a recent iteration of the testing suite). There might be easy fixes in there.

These are definitely issues with the `revngABI`, testing binary output looks correct.

##### `rft-insufficiency.yml`

These tests are RFT-specific. That is to say, something about the difference between the `CFT->RFT` path and the `CFT->Layout` path breaks these.
