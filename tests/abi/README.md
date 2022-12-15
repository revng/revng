# ABI Testing Utilities

## The testing pipeline.

The order of the operations done by the pipeline can be found in `test.sh` script that describes it. Let me elaborate a bit further on every command:

### `revng model import dwarf`

The abi testing reference binary is imported into a model using the normal means. The code for generating this testing artifact is located in the [revng-qa](https://github.com/revng/revng-qa) repository. Let's call the resulting model `reference_binary.yml`

### `revng model opt --convert-all-cabi-functions-to-raw`

This model opt pass converts every CABI function within the input model (`reference_binary.yml`) into its "raw" representation. The result is saved into (`downgraded_reference_binary.yml`)

### `revng model opt --convert-all-raw-functions-to-cabi`

This pass tries to "upgrade" every "raw" function into its CABI representation knowing the ABI it uses. The result is written to `upgraded_downgraded_reference_binary.yml`.

### `revng model opt --convert-all-cabi-functions-to-raw`

The same pass is run again to obtain one more set of "raw" functions. Those get written into `downgraded_upgraded_downgraded_reference_binary.yml`.

### `revng ensure-rft-equivalence`

This tool is used to compare `downgraded_reference_binary.yml` and `downgraded_upgraded_downgraded_reference_binary.yml`. If no information was lost during these conversions, the models are to be the same. More [here](##RFT-equivalence-testing).

### `revng check-compatibility-with-abi`

As the last step, normal abi verification is run on every single step of the conversion. This makes sure that at no point during the test, a "written" model was incompatible with the real binary it's supposed to represent. More [here](##How-is-the-artifact-used).

## The structure of the artifact file.

To test the ABI conformity, a helper generator (located in [revng-qa](https://github.com/revng/revng-qa)) is used. The output of the generated analysis helpers is further referred as an 'artifact'.

It's formatted as a YAML document. The "analysis" binary writes it to `stdout` when run. Current cmake-based build system puts this file under

```
${ORCHESTRA_ROOT}/share/revng/qa/tests/abi/${TESTED_ABI_NAME}/${TESTED_ARCHITECTURE_NAME}/compiled-run/analyzed-binary/default.stdout
```

The "clean" version of the binary (can be used to extract dwarf information, lift, etc.) is named `reference-binary`.

Here's the structure of the artifact itself:

```yml
---
- Stack:
    - Offset: ${STACK_OFFSET}
      Value: ${VALUE_AT_THE_OFFSET}
    # . . .
  Registers:
    - Name: ${REGISTER_NAME}
      VALUE: ${REGISTER_VALUE}
    # . . .
  Function: "${FUNCTION_NAME}"
  IsLittleEndian: "${YES_OR_NO}"
  Arguments:
    - Type: "${ARGUMENT_TYPE}"
      Address: "${ARGUMENT_ADDRESS}"
      Value: "${ARGUMENT_VALUE}"
    # . . .
  ReturnValue:
    - Type: "${RETURN_VALUE_TYPE}"
      Address: "${RETURN_VALUE_ADDRESS}"
      Value: "${RETURN_VALUE_VALUE}"
...
```

The utilities for parsing the artifact (rely on `llvm::yaml`) can be found in the sources of `revng-abi-verify` tool.

## How is the artifact used

The artifact provides information that can be used to verify whether a given argument COULD have been passed using a given register. There's no certainty: since all the values are randomly generated, collisions are likely. That's why each function has multiple iterations per artifact. Also that means that artifact is ineffective for restoring the original function signature, but it's good enough to verify whether a known signature corresponds to the factual state.

`revng check-compatibility-with-abi` tool goes through every function present in both the artifact and the model it was passed and and checks the conformity of the data within the model. For example, if the function within the model was fiddled with (its type was changed, for example) and was changed in a non-backwards-compatible way, this could be a good way to detect the problem.

## RFT equivalence testing

Sadly, there are cases where `revng model diff` is not good enough to verify that two instances of a model are the same. Mostly because it relies on the type IDs. So even if there are two identical types generated in different places (they have different IDs because of that), the diff detects these "changes" and fails the check.
`revng ensure-rft-equivalence` is a specialized extension of `revng model diff` that, on top of normal diff behavior, allows two identical RFT have different `StackArgumentsType` structs as long as they are identical in terms of everything but their IDs. This workaround was forced by the specifics of the testing pipeline, where there is no way to generated these `StackArgumentsType` structs with a known ID.
