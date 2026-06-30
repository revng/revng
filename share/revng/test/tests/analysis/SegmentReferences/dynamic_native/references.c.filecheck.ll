;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

CHECK: define i64 @local_main() [[IGN:.*]] {

; Ensure we detected a plain segment reference
CHECK: call i64 @local_[[IGN:.*]](i64 add (i64 ptrtoint (ptr @segment_[[IGN:.*]] to i64), i64 [[IGN:[0-9]*]]))

; Detect function pointer
CHECK: call i64 @local_[[IGN:.*]](i64 ptrtoint (ptr @local_[[IGN:.*]] to i64))

; Detect pointer to ASCII string
CHECK: call i64 @local_[[IGN:.*]](i64 ptrtoint (ptr @revng.const.ascii-string to i64))

; Detect pointer to non-ASCII (but still UTF8) string
CHECK: call i64 @local_[[IGN:.*]](i64 ptrtoint (ptr @revng.const.ff3ed8759b210a5e8ebd9328236874143571bc39 to i64))

; Detect pointer to UTF-16 string
CHECK: call i64 @local_[[IGN:.*]](i64 ptrtoint (ptr @revng.const.fd8fb4911c20096b606d41c5d40c4c8a8dd6fa54 to i64))

CHECK: }
