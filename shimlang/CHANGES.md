# Shimlang Fixes & Documentation Plan

Tracking document for changes requested after the documentation-completeness
exploration. Checkboxes track progress.

## A. Behavior changes (implementation)

- [x] **A1. Division by zero returns 0.** `x / 0` (and float `x / 0.0`) should
  evaluate to `0`, not `inf`/`NaN`.
- [x] **A2. Modulo by zero returns 0.** `x % 0` should evaluate to `0` instead
  of panicking (currently an uncaught Rust `rem_euclid` panic).
- [x] **A3. Integer overflow/underflow saturates.** Arithmetic that would
  overflow/underflow an `i32` (`+`, `-`, `*`, `pow`, …) should saturate at
  `i32::MIN`/`i32::MAX` instead of panicking.
- [x] **A4. Recursive formatting shows `...`.** Printing/formatting a value that
  is already being formatted (cyclic list/struct/etc.) must emit `...` for the
  repeated value instead of recursing into a stack overflow.
- [x] **A5. `break`/`continue` outside a loop is a parse/compile error.**
  Currently panics at runtime ("break should have loop info"). Detect at the
  parse/compile stage and return an `Err`.
- [x] **A6. Parsing fails for non-ASCII string literals.** A string literal
  containing non-ASCII bytes should be a parse error (today it parses and then
  crashes on indexing/iteration with a `FromUtf8Error` panic).
- [x] **A7. `bool` is a built-in.** Add `bool(value)` converting truthiness to a
  boolean (documented but currently `Unknown identifier`).
- [x] **A8. Chained comparisons.** Support `a < b < c` style chains.
  - `lower() < thing.value() < upper()` evaluates `thing.value()` exactly once.
  - `upper()` is only evaluated if `lower() < thing.value()` is true.
  - Applies to the chain of comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`).
- [x] **A9. Float literal exponents.** Support `e`/`E` exponent notation in float
  literals (e.g. `1.0e20`, `1.5E-3`).
- [x] **A10. Digit separators.** Support `_` separators in int and float
  literals (e.g. `1_000`, `1_000.000_1`).
- [x] **A11. Distinct dict keys for `1` and `1.0`.** Integer and float keys that
  are numerically equal should be distinct dictionary keys.
  - NaN key being unfindable is acceptable — leave as-is.
- [x] **A12. Leading-dot floats.** Support `.5` as a float literal (== `0.5`).
- [x] **A13. Non-finite float formatting drops `.0`.** Print `inf`, `-inf`,
  `NaN` (no trailing `.0`).
- [x] **A14. Assertion message uses string formatting.** `assert` failure output
  should format the value via the normal string-formatting path, not the
  internal debug representation.
- [x] **A15. Never surface internal representations.** No user-facing error or
  output should show internal debug forms (e.g. `Identifier [120]`,
  `Bool(false)`, `Integer(0)`, `parse_primary`). Use source names / formatted
  values / friendly wording everywhere.

## B. Things intentionally left as-is (confirm + document)

- [ ] **B1. Tuples have no `.len()` and no negative indexing.** They are
  fixed-size, heterogeneous (pair/triple/…), Rust-like rather than Python-like.
  Keep current behavior; document it.
- [ ] **B2. NaN dict key is not findable.** Acceptable; document under
  dictionaries.
- [ ] **B3. Non-symmetric operator overloading is fine.** Overloading is meant
  to operate on the same types. Keep; document the expectation.

## C. Documentation updates (LANGUAGE.md)

- [ ] **C1. Bools are not numbers.** State explicitly that `true`/`false` are not
  numeric: no arithmetic, and `1 != true`. (`int(true)`/`float(true)` still
  convert.)
- [ ] **C2. Document the second `assert` argument.** `assert(condition, message)`
  — message shown on failure.
- [ ] **C3. Range endpoints can be any (ordered) value.** Clarify that `..` /
  `Range()` endpoints aren't limited to ints (floats work; step semantics).
- [ ] **C4. Closures capture a fresh per-iteration binding.** Document loop-var
  capture semantics with an example.
- [ ] **C5. Mutable default args are re-evaluated each call.** Document that
  default expressions run on every call where the arg is omitted (no shared
  mutable default).
- [ ] **C6. Comparison overloads coerce the result by truthiness.** Document that
  the value returned from a comparison overload is interpreted via truthiness.
- [ ] **C7. Chained comparisons.** Document the new chaining semantics (A8),
  including single-evaluation and short-circuit behavior.
- [ ] **C8. Numeric literal syntax.** Document `e`/`E` exponents, `_` separators,
  and leading-dot floats (A9, A10, A12).
- [ ] **C9. Dict keys `1` vs `1.0` are distinct** (A11); NaN key note (B2).
- [ ] **C10. `bool` built-in.** Already listed; ensure it's accurate after A7.
- [ ] **C11. Division/modulo by zero return 0; integer overflow saturates.**
  Document these as defined behaviors (A1–A3).
- [ ] **C12. Tuples are heterogeneous fixed-size values** (B1): no `.len()`, no
  negative indexing, not iterable; primarily for grouping/unpacking.
- [ ] **C13. Non-finite floats render as `inf`/`-inf`/`NaN`** (A13).

## Working order

1. Documentation-only items that reflect already-correct behavior (C1, C2, C4,
   C5, C6, B-section docs) can land early.
2. Implementation items, each with a focused test via `cargo run --bin shm`.
3. Documentation for new/changed behavior (C3, C7, C8, C9, C11, C13) alongside
   or right after the corresponding implementation change.
