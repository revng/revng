# The end to end example of what happens within the testing suite

Note that this is a continuation, for the beginning of the explanation, see the [`scripts/abi_artifact_generator/end_to_end.md`](https://github.com/revng/revng-qa/blob/develop/scripts/abi_artifact_generator/end_to_end.md) in the `revng-qa` repository.

The explanation on that side ends with us having successfully produced three distinct binaries:
* `"functions binary"` - the file containing debug information used for importing test function prototypes into a model.
* `"mmap'ed binary"` - the file containing runnable test functions as well as setup they require.
* `"runner binary"` - the binary that accepts a path to the `"mmap'ed binary"` as an argument and contains the infrastructure to run relevant pieces of it one test at a time while injecting interrupts to extract CPU state at certain predetermined points of execution.

The testing starts with running the `runner`. For `i386` and `x86_64` it's done natively, while all the other architectures use QEMU. As the result of the execution, a yaml fine describing the test progress is produced. You can read further about how it's structured [here](https://github.com/revng/revng-qa/blob/develop/scripts/abi_artifact_generator/output_format.md)

For the current example, it would look something like for the argument tests:
```yaml
TargetArchitecture: x86_64
IsLittleEndian: yes
Iterations:
# only a single hand-picked iteration is shown to keep the example brief.
  - Function: "args"
    Iteration: "0"
    Arguments:
      - Type: small_struct # size = 4
        Address: 0x00007fff6fb34678
        AddressBytes: [ 0x78, 0x46, 0xb3, 0x6f, 0xff, 0x7f, 0x00, 0x00 ]
        ExpectedBytes: [ 0xc3, 0x53, 0xe3, 0xe1 ]
        FoundBytes: [ 0xc3, 0x53, 0xe3, 0xe1 ]
      - Type: big_struct # size = 64
        Address: 0x00007fff6fb34674
        AddressBytes: [ 0x74, 0x46, 0xb3, 0x6f, 0xff, 0x7f, 0x00, 0x00 ]
        ExpectedBytes: [ 0x53, 0xe5, 0xf1, 0xbc, 0x43, 0xad, 0xef, 0x78, 0xfd, 0x8b, 0x44, 0x31, 0xf1, 0x3d, 0x4c, 0xe2, 0x01, 0x67, 0xf3, 0xd4, 0x84, 0x11, 0xb5, 0x89, 0x40, 0x6a, 0x45, 0x5f, 0xa3, 0xcc, 0xb7, 0x66, 0xb0, 0xef, 0x73, 0x78, 0x79, 0x15, 0xf6, 0x2a, 0x34, 0x0c, 0x25, 0xe2, 0xe2, 0x8f, 0xc6, 0x1f, 0x17, 0xc5, 0x9b, 0xc9, 0x09, 0xe4, 0x12, 0x48, 0xce, 0x13, 0xab, 0x96, 0x06, 0xcb, 0x0d, 0xb6 ]
        FoundBytes: [ 0x53, 0xe5, 0xf1, 0xbc, 0x43, 0xad, 0xef, 0x78, 0xfd, 0x8b, 0x44, 0x31, 0xf1, 0x3d, 0x4c, 0xe2, 0x01, 0x67, 0xf3, 0xd4, 0x84, 0x11, 0xb5, 0x89, 0x40, 0x6a, 0x45, 0x5f, 0xa3, 0xcc, 0xb7, 0x66, 0xb0, 0xef, 0x73, 0x78, 0x79, 0x15, 0xf6, 0x2a, 0x34, 0x0c, 0x25, 0xe2, 0xe2, 0x8f, 0xc6, 0x1f, 0x17, 0xc5, 0x9b, 0xc9, 0x09, 0xe4, 0x12, 0x48, 0xce, 0x13, 0xab, 0x96, 0x06, 0xcb, 0x0d, 0xb6 ]
StateBeforeTheCall:
      Registers:
        - Name: "rdi"
          Value: 0x7d994864e1e353c3
          Bytes: [ 0xc3, 0x53, 0xe3, 0xe1, 0x64, 0x48, 0x99, 0x7d ]
# other registers are omitted as they are irrelevant for this example.
      Stack: [ 0x53, 0xe5, 0xf1, 0xbc, 0x43, 0xad, 0xef, 0x78, 0xfd, 0x8b, 0x44, 0x31, 0xf1, 0x3d, 0x4c, 0xe2, 0x01, 0x67, 0xf3, 0xd4, 0x84, 0x11, 0xb5, 0x89, 0x40, 0x6a, 0x45, 0x5f, 0xa3, 0xcc, 0xb7, 0x66, 0xb0, 0xef, 0x73, 0x78, 0x79, 0x15, 0xf6, 0x2a, 0x34, 0x0c, 0x25, 0xe2, 0xe2, 0x8f, 0xc6, 0x1f, 0x17, 0xc5, 0x9b, 0xc9, 0x09, 0xe4, 0x12, 0x48, 0xce, 0x13, 0xab, 0x96, 0x06, 0xcb, 0x0d, 0xb6, 0x14, 0x4d, 0xff, 0xee, 0x42, 0x97, 0x85, 0x76 ] # the rest of the stack is omitted.
# other states, namely `StateAfterTheCall` and `StateAfterTheReturn` are omitted.
```
Even just looking at this file as is, it's very easy to see that the first argument (4-byte struct) is placed in the lower half of `rdi` register, while the second one (64-byte struct) - on the piece of the stack starting from the stack pointer.

Same can be said about the return values as well, here's the small struct example:
```yaml
TargetArchitecture: x86_64
IsLittleEndian: yes
Iterations:
# only a single hand-picked iteration is show to keep the example brief.
  - Function: "small_rv"
    Iteration: "0"
    ReturnValues:
      - Type: small_struct # size = 4
        Address: 0x00007fff6fb34694
        AddressBytes: [ 0x94, 0x46, 0xb3, 0x6f, 0xff, 0x7f, 0x00, 0x00 ]
        ExpectedBytes: [ 0x5d, 0xff, 0x0f, 0x0e ]
        FoundBytes: [ 0x5d, 0xff, 0x0f, 0x0e ]
StateAfterTheReturn:
      Registers:
        - Name: "rax"
          Value: 0x000000000e0fff5d
          Bytes: [ 0x5d, 0xff, 0x0f, 0x0e, 0x00, 0x00, 0x00, 0x00 ]
# other registers are omitted as they are irrelevant for this example.
      Stack: [] # the stack is omitted.
# other states, namely `StateBeforeTheCall` and `StateAfterTheCall` are omitted.
```
since the return value is small, we can easily find it in the `rax` register - so no need to look into the stack or `ReturnValueAddress`.
NOTE: notice how the top half of rax is empty - this is a really good indication of the fact that compiler used a 4-byte `mov` to put it there since our return value also happens to be just 4 bytes.

As for the big return value example, it's slightly more complex:
```yaml
TargetArchitecture: x86_64
IsLittleEndian: yes
Iterations:
# only a single hand-picked iteration is show to keep the example brief.
  - Function: "big_rv"
    Iteration: "0"
    ReturnValues:
      - Type: big_struct # size = 64
        Address: 0x00007ffc21d6f298
        AddressBytes: [ 0x98, 0xf2, 0xd6, 0x21, 0xfc, 0x7f, 0x00, 0x00 ]
        ExpectedBytes: [ 0xe7, 0x7d, 0xb5, 0xe9, 0xcb, 0xec, 0x82, 0x3c, 0x8a, 0xe1, 0xb7, 0x0e, 0x57, 0xcd, 0x61, 0x91, 0xa7, 0x08, 0xf6, 0x44, 0xfe, 0x95, 0xe8, 0x2c, 0x7a, 0x86, 0x46, 0xb3, 0x80, 0x6f, 0x4e, 0x9d, 0xa3, 0xe2, 0x72, 0x75, 0x20, 0xac, 0xf4, 0x29, 0xf9, 0xc9, 0xe5, 0x27, 0x18, 0x7d, 0x87, 0xde, 0x82, 0x96, 0x0b, 0x1a, 0xca, 0x61, 0xe2, 0xd8, 0xe1, 0x2e, 0x87, 0x8b, 0x97, 0xa8, 0x49, 0x9a ]
        FoundBytes: [ 0xe7, 0x7d, 0xb5, 0xe9, 0xcb, 0xec, 0x82, 0x3c, 0x8a, 0xe1, 0xb7, 0x0e, 0x57, 0xcd, 0x61, 0x91, 0xa7, 0x08, 0xf6, 0x44, 0xfe, 0x95, 0xe8, 0x2c, 0x7a, 0x86, 0x46, 0xb3, 0x80, 0x6f, 0x4e, 0x9d, 0xa3, 0xe2, 0x72, 0x75, 0x20, 0xac, 0xf4, 0x29, 0xf9, 0xc9, 0xe5, 0x27, 0x18, 0x7d, 0x87, 0xde, 0x82, 0x96, 0x0b, 0x1a, 0xca, 0x61, 0xe2, 0xd8, 0xe1, 0x2e, 0x87, 0x8b, 0x97, 0xa8, 0x49, 0x9a ]
StateBeforeTheCall:
      Registers:
        - Name: "rdi"
          Value: 0x00007ffc21d6f298
          Bytes: [ 0x98, 0xf2, 0xd6, 0x21, 0xfc, 0x7f, 0x00, 0x00 ]
# other registers and the stack are omitted as they are irrelevant for this example.
StateAfterTheReturn:
      Registers:
        - Name: "rax"
          Value: 0x00007ffc21d6f298
          Bytes: [ 0x98, 0xf2, 0xd6, 0x21, 0xfc, 0x7f, 0x00, 0x00 ]
        - Name: "rsp"
          Value: 0x00007ffc21d6f298
          Bytes: [ 0x98, 0xf2, 0xd6, 0x21, 0xfc, 0x7f, 0x00, 0x00 ]
# other registers are omitted as they are irrelevant for this example.
      Stack: [ 0xe7, 0x7d, 0xb5, 0xe9, 0xcb, 0xec, 0x82, 0x3c, 0x8a, 0xe1, 0xb7, 0x0e, 0x57, 0xcd, 0x61, 0x91, 0xa7, 0x08, 0xf6, 0x44, 0xfe, 0x95, 0xe8, 0x2c, 0x7a, 0x86, 0x46, 0xb3, 0x80, 0x6f, 0x4e, 0x9d, 0xa3, 0xe2, 0x72, 0x75, 0x20, 0xac, 0xf4, 0x29, 0xf9, 0xc9, 0xe5, 0x27, 0x18, 0x7d, 0x87, 0xde, 0x82, 0x96, 0x0b, 0x1a, 0xca, 0x61, 0xe2, 0xd8, 0xe1, 0x2e, 0x87, 0x8b, 0x97, 0xa8, 0x49, 0x9a, 0xdc, 0x12, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00 ] # the rest of the stack is omitted.
```
Here, we actually have to look at the `ReturnValueAddress` first. And we do indeed find it in both `rdi` before the call and `rax` after the call. This indicates that the callee is the one reserving the memory to put the return value in and used `rdi` register to indicate where that memory is. And also that `rax` is used to return the same pointer back.
After we verified that (if we make the assumption, that a stack location is always used), we need to find the offset between the current stack pointer and the pointer to return value - so that we can confirm the bytes of the return value itself. Luckily, in this case (as you can see above) they match - meaning that the return value is on the top of the stack right after the return. What's left is to just verify that the bytes themselves match the expectations as well.

But the logic above is highly subjective (look here, look there, etc.). To make testing efficient, we need to put it into more rigid terms: which is what `test-X.sh` scripts do. For more details on those see the [README](./README.md).

Here's an example of the logic used to test the compatibility of an arbitrary model prototype with the register data we obtained using the methods above:

First of all, to avoid relying on a specific function type (Raw or CABI), an `abi::FunctionType::Layout` helper is used. See the [definition](../../../../../include/revng/ABI/FunctionType/Layout.h).
For our `test_args`, the prototype looks like
```yaml
  - ID:              1
    Kind:            CABIFunctionType
    ABI:             SystemV_x86_64
    ReturnType:
      UnqualifiedType: "/Types/256-PrimitiveType" # void
    Arguments:
      - Index:           0
        Type:
          UnqualifiedType: "/Types/2-StructType"
        CustomName:      "argument_0"
        OriginalName:    argument_0
      - Index:           1
        Type:
          UnqualifiedType: "/Types/3-StructType"
        CustomName:      "argument_1"
        OriginalName:    argument_1
  - ID:              2
    Kind:            StructType
    Size:            4
    Fields:
      - Offset:          0
        CustomName:      "a"
        OriginalName:    a
        Type:
          UnqualifiedType: "/Types/1282-PrimitiveType"
      - Offset:          2
        CustomName:      "b"
        OriginalName:    b
        Type:
          UnqualifiedType: "/Types/1282-PrimitiveType"
  - ID:              3
    Kind:            StructType
    Size:            64
    Fields:
      - Offset:          0
        CustomName:      "a"
        OriginalName:    a
        Type:
          UnqualifiedType: "/Types/1288-PrimitiveType"
          Qualifiers:
            - Kind:            Array
              Size:            8
  - ID:              1282
    Kind:            PrimitiveType
    PrimitiveKind:   Unsigned
    Size:            2
  - ID:              1288
    Kind:            PrimitiveType
    PrimitiveKind:   Unsigned
    Size:            8
```
But just looking at it does give much insight about how those arguments are passed in. Which is where layout comes in.
```yaml
# test_args

# some fields (like `Type`) are omitted for the sake of brevity
Arguments:
  - Registers: ["rdi"]
    StackSpan: no
    Kind: Scalar
  - Registers: []
    StackSpan: { Offset: 0, Size: 64 }
    Kind: ReferenceToAggregate
ReturnValues:
    # empty
```
Which, if you remember the previous step, exactly matches the conclusion we came up with while looking at the yaml output. So, for the test, the only thing that's left is to go through the layout and, for each argument, make sure that the value extracted from the location layout outputs exactly matches that of the expected argument value.

(from now on, I'll be omitting model representation of the types:  they are very verbose)

The small return value is very similar:
```yaml
# test_small_rv

# some fields (like `Type`) are omitted for the sake of brevity
Arguments:
  # empty
ReturnValues:
  - Registers: ["rax"]
```
All that needs checking is the value of `rax` after the return.

The big return value case, on the other hand, is a bit more complex:
```yaml
# test_big_rv

# some fields (like `Type`) are omitted for the sake of brevity
Arguments:
  - Registers: ["rdi"]
    StackSpan: no
    Kind: ShadowPointerToAggregateReturnValue
ReturnValues:
  - Registers: ["rax"]
```
As you can see, we had an argument appear in the mix. So, first, we need to check that argument as usual: as in, we verify that the value of `ReturnValueAddress` indeed matches whatever is in `rdi` _before_ the call.
After that we also check return value as usual: as in, we verify that the value of `ReturnValueAddress` also matches that of `rax` _after_ the return.

But also notice, that it has a special `Kind`: to indicate that it's actually a return value in disguise. Looking at the kind is enough of a hint that we should do additional checks - those related to the returned bytes. We assume that such return values always use stack - so we find an offset between the value of the stack pointer and the pointer to the return value - then we use said offset to check bytes against the expectations.

If at any point any expectation doesn't match - we assume layout is wrong and fail the test.
If, on the other hand, this test is passed for all 4 models, we can go to the final check - to ensure converting function back and force doesn't break it.

By definition, the `raw->cabi` conversion is not always possible (can fail because any arbitrary function is not guaranteed to match any specific ABI), but if it doesn't - no information is lost - the conversion is perfect.
On the other hand, the `cabi->raw` conversion is always successful (no matter the ABI, a function can be represented as a set of registers marked as arguments and return values, as well as some description of its stack frame) - but because of limited expressibility of `RawFunctionType`, some type information is bound to be lost. Because of that, this conversion is _lossy_.

So, in order to verify that conversion loop is stable despite it having a lossy conversion present, we need to first loose all the information we cannot represent as an RFT. Luckily, that's super simple - the converter does that for free :)

So, if we want to verify a conversion loop to not change anything, we just need to do a single `cabi->raw` conversion (note that we start from the `cabi` state since that's what analyzing debug information gives us). For better understanding of the process (and the graph), see the [README](./README.md) related to the tool doing the comparison.

So, a short summary to end the example:
1. Produce a binary that for each _instrumented_ function call dumps all the values of arguments, return values as well as the CPU state.
2. Chain multiple function type conversions back and forth.
3. Verify that at no point there is a tested function that contradicts the output of the binary from step 1.
4. Verify that post-lossy conversion, no matter how many cycles are done, the prototypes stay the same (don't degrade any further).

