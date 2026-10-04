# Privileged setup filesystem boundary

Implementation tracker: #81 (release review #79, finding F4).

## Required separation

Runtime data is writable by the Worker. Installer completion receipts are not
runtime data and must live in a separate root-owned directory. Legacy receipts
under the Worker home/state must not become trusted setup evidence.

An installer receipt records that a reviewed action completed; it is not proof
that the runtime executable or model still exists. Repeat setup must inspect
the current prerequisite, rather than accept marker presence alone.

## Filesystem contract

Privileged setup must refuse symlinks (including dangling links), unsafe
ownership/permissions and unexpected file types at authoritative locations.
Directory traversal and atomic writes must be tied to verified directory file
descriptors rather than a check followed by an unprotected pathname write.
Unrelated targets must remain unchanged when validation fails.

Existing unsafe objects must not be silently adopted by changing their owner
or mode. Recovery must be an explicit operator decision described by the final
implementation documentation.

## Release boundary

This document establishes the intended contract, not acceptance evidence.
The implementation and its adversarial regression tests are in progress in
#81. #79 remains HOLD. #80 is a separate documentation-only real-hardware
installation and operation gate whose acceptance authority is upiscium.
