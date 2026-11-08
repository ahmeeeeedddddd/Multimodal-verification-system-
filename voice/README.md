# Voice Module (Task 3 + Task 4)

Speaker verification and a verification-gated speech-command system.

## 1. Overview

| Part | What it does |
|---|---|
| **Task 3** | Voice-only verification: given an audio clip and a claimed identity, output a genuine probability and accept/reject using a threshold. |
| **Task 4** | Isolated-word command recognizer (on, off, go, stop, yes, no) plus a permission-gated executor tied to the verified identity. |

## 2. Folder layout

```
voice/
├── src/
│   ├── config.py            # paths, sample rate, trimming threshold, seeds
│   ├── preprocessing.py     # load -> trim silence -> normalize -> fix length
│   ├── features.py          # extract_voice_features (30-D) - the fusion interface
│   ├── dataset.py           # speaker lists, session-aware splits, trial generation
│   ├── train_voice.py       # trains + tunes SVM and Random Forest, saves models
│   ├── verify_voice.py      # VoiceVerifier API
│   ├── metrics_basic.py     # EER / FAR / FRR / AUC (replaced by the shared module in Task 6)
│   ├── record_team.py       # record team voice clips (Bonus 3)
│   ├── record_live.py       # record a live test session (honest validation/test)
│   ├── live_mic.py          # live microphone verification (Bonus 2)
│   ├── live_diag.py         # shows the full probability vector for debugging
│   ├── commands.py          # command recognizer: training + CommandRecognizer
│   ├── permissions.py       # permission table + executor
│   ├── live_commands.py     # live demo: digit for identity, then a command word
│   └── test_*.py            # sanity checks
└── models/
    ├── voice_svm.joblib     # speaker verification (SVM)
    ├── voice_rf.joblib      # speaker verification (Random Forest)
    └── command_svm.joblib   # command recognizer
```

## 3. Setup

```
git clone https://github.com/ahmeeeeedddddd/Multimodal-verification-system-.git
cd Multimodal-verification-system-
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd voice\src
```

Always activate the venv in every new terminal. Run all voice scripts from `voice\src`.

## 4. Data (not stored in Git)

Put these under `data/` (it is in `.gitignore`):

| Folder | Source | Used for |
|---|---|---|
| `data/fsdd/recordings/` | Free Spoken Digit Dataset (6 speakers, 8 kHz) | Task 3 |
| `data/team_voice/` | `record_team.py` (team members, `digit_name_index.wav`) | Task 3 enrollment |
| `data/live_test/` | `record_live.py` (live session, same naming) | Task 3 val/test |
| `data/speech_commands/` | Google Speech Commands (16 kHz) | Task 4 |

## 5. Task 3: Speaker verification

### Pipeline
1. **Preprocess** (`preprocessing.py`): mono 8 kHz, trim silence (`TOP_DB = 15`), remove DC offset, peak-normalize, pad/crop to 1 s (8000 samples). Normalization happens before padding so the padding stays exactly zero.
2. **Features** (`features.py`): 30-D vector = 13 MFCC mean + 13 MFCC std + ZCR (mean, std) + RMS (mean, std).
3. **Model**: one closed-set multiclass classifier (StandardScaler + SVM or Random Forest) over the enrolled speakers.
4. **Score**: `genuine_probability = P(claimed identity | audio)`. Accept if `probability >= threshold`.

### Identities
- **Enrolled (trained on):** FSDD `george, jackson, lucas, nicolas` + every team member found in `data/team_voice/`.
- **Unknown (never trained on):** FSDD `theo, yweweler`, used only to build unknown-impostor trials.

### Trials
- **Genuine:** a recording claims its own identity.
- **Known impostor:** an enrolled speaker claims another enrolled identity.
- **Unknown impostor:** an unknown speaker claims an enrolled identity.

### Split strategy (session-aware)
- FSDD enrolled speakers: random split stratified by (speaker, digit): 60/20/20 train/val/test.
- Team members: `data/team_voice` -> **train only**; `data/live_test` -> split 50/50 into **val** and **test**.
- The decision threshold is chosen at the EER on **val** and applied unchanged to **test**.

A purely random split of one recording session gave over-optimistic numbers (the model partly learned session/microphone conditions). The session-aware split was introduced after a live microphone test exposed this.

### Training
```
python -W ignore::FutureWarning train_voice.py
```
Models are tuned with 5-fold GridSearchCV on the training set only.

### Results (latest run, 6 enrolled identities)

| Model | TEST AUC | TEST EER | FAR | FRR | Unknown impostors accepted |
|---|---|---|---|---|---|
| **SVM** | 0.9967 | 3.06% | 1.88% | 4.40% | 3.6% |
| RF | 0.9952 | 3.65% | 4.08% | 2.86% | 7.8% |

Known impostors accepted: 0% for both. `<update after the final retrain>`

### Using the verifier
```python
from verify_voice import VoiceVerifier
v = VoiceVerifier("svm")                               # or "rf"
r = v.verify(raw_audio_8khz, claimed_id="basmala")
# {"claimed": ..., "probability": ..., "accepted": ..., "threshold": ...}
```

### Feature interface for fusion (Task 5)
```python
from preprocessing import preprocess
from features import extract_voice_features, VOICE_FEATURE_DIM   # 30
vec = extract_voice_features(preprocess(raw_audio_8khz))          # 1-D float32
```
Input: mono waveform at 8000 Hz. Standardize the vector separately before concatenating it with other modalities.

### Live microphone (Bonus 2)
```
python live_mic.py --claim <name> --rounds 10
```
Uses the same `preprocess` and `extract_voice_features` code as training.

### Recording team data (Bonus 3)
```
python record_team.py --name <yourname> --reps 15      # run twice, with a break between
python record_live.py --name <yourname> --reps 10      # live session, ideally on the demo laptop
```
Names must be lowercase English with no spaces or underscores. Set the Windows input volume to about 50-60% to avoid clipping.

## 6. Task 4: Speech commands

### Recognizer (`commands.py`)
- Runs at 16 kHz. Words: `on, off, go, stop, yes, no`, plus an extra **`unknown`** class (cat, dog, bed, bird, happy, house, tree, wow) so non-commands are rejected instead of forced into a command.
- Features (82-D): MFCC mean over 4 temporal segments, MFCC std, mean |delta MFCC|, ZCR and RMS statistics. Keeping temporal order raised accuracy by about 11 points over plain mean/std.
- Model: StandardScaler + SVM, tuned with GridSearchCV.
- Split: the dataset's official `validation_list.txt` / `testing_list.txt` (speaker-disjoint).

| Setup | TEST accuracy |
|---|---|
| 6 words, mean/std features | 76.4% |
| 6 words, time-aware features | 87.7% |
| 6 words + unknown class (final) | 85.2% |

Training: `python commands.py`

### Permission table (`permissions.py`)

| Word | Command | admin | user | unverified |
|---|---|---|---|---|
| on | lights_on | yes | yes | no |
| off | lights_off | yes | yes | no |
| go | status | yes | yes | no |
| stop | stop | yes | yes | no |
| yes | unlock | yes | no | no |
| no | delete | yes | no | no |

Roles come from `USER_ROLES` in `permissions.py`. A verified identity that is not listed defaults to `user`.

### Executor
```python
from commands import CommandRecognizer
from permissions import execute
word, conf = CommandRecognizer().recognize(audio_16khz)
result = execute(word, identity, verified)   # identity/verified come from the pipeline
```
The executor never verifies identity itself: it refuses protected actions unless the pipeline reports a verified identity with sufficient permission.

### Live demo
```
python live_commands.py --claim <name> --rounds 5
```
Each round: say a **digit** (identity check) and then a **command word**.

## 7. Known limitations

- Verification uses spoken **digits** (text-dependent), while commands are different words, so the demo does identity check first and the command second.
- FSDD speakers were recorded with different equipment than the team, so low FAR against them may partly reflect microphone differences, not only voice differences.
- Only 2 FSDD speakers act as unknown impostors, so the unknown-impostor rate is not statistically precise.
- The live val/test sets are small (55 clips each, one speaker so far), so FRR estimates are rough.
- The command recognizer is weaker on `go` vs `no`, and on non-studio microphones.
- Team members without a `live_test` session are trained on but not evaluated.

## 8. Status / TODO

- [x] Task 3: preprocessing, features, SVM + RF, threshold, verifier API, saved models
- [x] Bonus 2: live microphone verification
- [x] Task 4: recognizer, permission table, executor, live commands
- [ ] Bonus 3: live sessions for all team members (`<mohamed>` pending)
- [ ] Add the third team member and retrain
- [ ] Re-measure FAR with real impostors on the demo laptop