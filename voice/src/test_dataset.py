"""Sanity check for splits and trial generation."""
from collections import Counter

from dataset import (enrolled_speakers, UNKNOWN_SPEAKERS, extract_all,
                     split_enrolled, split_unknown, make_trials)

enrolled_ids = enrolled_speakers()
print("Enrolled identities:", enrolled_ids)
enrolled = extract_all(enrolled_ids)
unknown = extract_all(UNKNOWN_SPEAKERS)

train, val, test = split_enrolled(enrolled)
unk_val, unk_test = split_unknown(unknown)

print(f"\nEnrolled total: {len(enrolled.X)}")
print(f"  train={len(train.X)}  val={len(val.X)}  test={len(test.X)}")
print(f"Unknown total: {len(unknown.X)}  val={len(unk_val.X)}  test={len(unk_test.X)}")

print("\nTrain speakers:", dict(Counter(train.speaker)))

X, claimed, y, kind = make_trials(test, unk_test)
print(f"\nTest trials: {len(y)}")
print("  by kind :", dict(Counter(kind)))
print("  genuine/impostor:", dict(Counter(y)))