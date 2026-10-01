# squeaktest

![squeaktest: deepfake voice detector](docs/assets/banner.jpg)

> Real curds squeak. Fakes don't.

**squeaktest** estimates whether speech in an audio clip is synthetic (AI-generated or voice-cloned). Instead of a yes/no verdict, it gives a probability score in one of three bands (*Likely synthetic*, *Uncertain*, *Likely genuine*) and shows which parts of the clip look suspicious. Its accuracy is measured on audio and generators it wasn't trained on, and all results are published, including the weak ones. Voice counts as biometric data, so uploads are processed in memory, never stored, and never logged. It runs locally through a CLI or Docker, or on the web.

## Status

Early development (Phase 0: research). Nothing is usable yet, and no accuracy claims are made until evaluation results are published here.
