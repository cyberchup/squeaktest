# SIEM event (proposal)

*Status: deferred to Phase 5 (decided 2026-10-01). This is an unreviewed proposal and nothing
here is implemented. The decisions at the end are still open. Until then, `squeaktest analyze
--json` gives the result fields without the SIEM-specific source and identity fields.*

`squeaktest analyze --jsonl` (step 7) will emit one event per analyzed recording, shaped for a
Log Analytics custom table such as `VoiceDeepfake_CL`. It follows promptbadger's conventions,
so both projects' events look alike in Sentinel:

- **PascalCase fields describe the event and where the audio came from.**
- **snake_case fields hold squeaktest's result.**

## Example: a spliced fake in a voicemail

```json
{
  "TimeGenerated": "2026-10-01T14:05:12+00:00",
  "EventType": "VoiceDeepfakeScan",
  "Source": "voicemail-gateway",
  "ContentType": "voicemail",
  "ContentId": "vm-20261001-0042",
  "User": "jane.doe@contoso.com",
  "ClaimedCaller": "+1 555 0100 (CFO)",
  "band": "likely_synthetic",
  "score": 0.91,
  "calibrated": false,
  "aggregation": "top3_mean",
  "suspicious_segments": [
    {"start_s": 10.0, "end_s": 14.0, "score": 0.97},
    {"start_s": 8.0, "end_s": 12.0, "score": 0.88},
    {"start_s": 12.0, "end_s": 16.0, "score": 0.86}
  ],
  "duration_s": 24.0,
  "speech_s": 21.7,
  "windows_scored": 11,
  "windows_skipped": 0,
  "audio_format": "m4a/aac",
  "audio_bytes": 391245,
  "audio_sha256": "5f1c9e…",
  "model": "antideepfake-mms-300m@7928e11",
  "scanner_version": "0.1.0"
}
```

## Example: too little speech to assess

```json
{
  "TimeGenerated": "2026-10-01T14:07:40+00:00",
  "EventType": "VoiceDeepfakeScan",
  "Source": "voicemail-gateway",
  "band": "not_assessed",
  "score": null,
  "note": "less than 1 s of speech; not enough to score",
  "calibrated": false,
  "aggregation": "top3_mean",
  "suspicious_segments": [],
  "duration_s": 3.2,
  "speech_s": 0.4,
  "windows_scored": 0,
  "windows_skipped": 0,
  "audio_format": "wav/pcm_mulaw",
  "audio_bytes": 51244,
  "audio_sha256": "a03d77…",
  "model": "antideepfake-mms-300m@7928e11",
  "scanner_version": "0.1.0"
}
```

## Fields

### Event and source (PascalCase)

| Field | Required | Meaning | Why it's there |
|---|---|---|---|
| `TimeGenerated` | yes | UTC time of the analysis | Sentinel's time column |
| `EventType` | yes | Always `VoiceDeepfakeScan` | Rules filter on it, as promptbadger's do on `PromptInjectionScan` |
| `Source` | yes | The system that submitted the audio (`voicemail-gateway`, `helpdesk-recorder`, `cli`) | Tells you which pipeline to look at, and maps to an application entity |
| `ContentType` | no | `voicemail`, `call_recording`, `voice_note` or `upload` | Different content types deserve different alert thresholds |
| `ContentId` | no | The source system's ID for the recording (message ID, call ID) | Lets an analyst pull the original recording from the system that holds it |
| `User` | no | Who received the audio, as a UPN | Joins to `IdentityInfo` and `SigninLogs`, and maps to an Account entity. In vishing, this is the person being targeted |
| `ClaimedCaller` | no | Who the caller claimed to be: caller ID and/or the name given | Shows who is being impersonated, so you can spot campaigns against one executive's identity |

### Result (snake_case)

| Field | Meaning | Why it's there |
|---|---|---|
| `band` | `likely_synthetic`, `uncertain`, `likely_genuine` or `not_assessed` | The field rules alert on. `not_assessed` is a band of its own, so "couldn't tell" is never confused with "likely genuine" (decision D5) |
| `score` | 0 to 1, higher means more likely synthetic; `null` when not assessed | Lets analysts tune their own thresholds and sort results |
| `calibrated` | `false` until Phase 2 calibrates the scores | Keeps an uncalibrated number from being read as a probability (D9) |
| `note` | Why there is no score, when there isn't one | Gives the reason for `not_assessed` |
| `aggregation` | How window scores became one score (`top3_mean`) | Results mean different things under different methods (D6), so the method is recorded with them |
| `suspicious_segments` | Up to 5 highest-scoring windows, with start, end and score | Tells the analyst where in the recording to listen. Capped so events stay small |
| `duration_s`, `speech_s` | Clip length, and seconds of detected activity | Short or mostly-silent clips are less reliable, so they can be filtered or down-weighted |
| `windows_scored`, `windows_skipped` | How many windows got a score, and how many were skipped as silence | Reveals clips where most of the audio was never scored |
| `audio_format` | Container and codec, such as `m4a/aac` | Phone codecs and heavy compression affect accuracy (research section 4.1) |
| `audio_bytes` | File size | Basic metadata, and useful for spotting oddities |
| `audio_sha256` | SHA-256 of the file's bytes | Correlation: one deepfake voicemail sent to 40 employees shows up as 40 events with the same hash |
| `model` | Model name and pinned commit | A model change changes scores; this lets you separate results from different versions |
| `scanner_version` | squeaktest version | Same reason, for squeaktest's own logic |

### What the event never contains

The event never holds audio, transcripts, filenames or file paths. Identity fields (`User`,
`ClaimedCaller`, `ContentId`) appear only when the caller passes them in. A filename can carry
personal data, such as "ceo-call-jane.wav", and the analyst can get the recording through
`ContentId` instead.

## How a rule would use it

An analytics rule for Phase 5, to show that the fields earn their place:

```kql
// Likely synthetic voice, grouped by recording to surface campaigns
// MITRE ATT&CK: T1566.004 (Spearphishing Voice), T1656 (Impersonation)
VoiceDeepfake_CL
| where TimeGenerated > ago(1h)
| where EventType == "VoiceDeepfakeScan" and band == "likely_synthetic"
| summarize
    Recipients = dcount(User),
    Users = make_set(User, 50),
    ClaimedCallers = make_set(ClaimedCaller, 10),
    MaxScore = max(score),
    FirstSeen = min(TimeGenerated)
    by audio_sha256, Source
| extend AlertSeverity = iff(Recipients >= 3, "High", "Medium")  // same recording, many targets
```

## Decisions for Dylan

1. **The privacy rule and identifiers.** The brief says to never log "raw scores tied to an
   identifier", but a SIEM event is only useful because it ties a result to a recording and a
   recipient. Proposal: that rule binds squeaktest's own logs and the hosted demo, which never
   emits events at all. SIEM events are opt-in output to the operator's own SIEM, and
   identifiers appear only when the operator passes them in. If you agree, I'll reword the
   brief to say exactly that.
2. **Score scale.** I propose 0 to 1. promptbadger uses 0 to 100, but its score is a points
   total from rules, while squeaktest's will be a calibrated probability, and 0 to 1 keeps that
   meaning visible. Switching to 0 to 100 for consistency between the two projects is a
   reasonable choice too.
3. **`ClaimedCaller`.** Keep it? It's what links a fake to the person being impersonated, but
   caller ID is easily spoofed, so it describes the claim, not the truth.
4. **`audio_sha256` by default.** It enables campaign correlation, and a hash reveals nothing
   about the voice. The trade-off is that the same recording becomes recognizable across
   events, by design. My recommendation is on by default, with an option to turn it off.
5. **Anything you'd want as a Sentinel analyst that's missing?** For example, a direction field
   (inbound or outbound call) or an ASIM-style normalization.
