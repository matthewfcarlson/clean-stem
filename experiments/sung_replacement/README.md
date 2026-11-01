# Sung replacement experiment

Can we replace a swear with a clean word sung in the artist's voice instead of
blanking it? Everything here runs outside the project's dependencies.

Pipeline per word:

1. `prep.py` (project venv): Demucs-separate a window around the word and a
   reference stretch of the artist's vocals.
2. `make_source.py`: speak the replacement with macOS `say`, then use WORLD to
   give it the swear's length and pitch contour.
3. Seed-VC (`inference.py`, singing model `--f0-condition True`) converts that
   source into the artist's voice using the reference.
4. `splice_variants.py`: render original / reversed / instrumental / Seed-VC
   (and optionally "borrowed": the same word sung elsewhere in the song,
   reshaped with WORLD) clips.

Setup (separate venv, Python 3.10):

    git clone --depth 1 https://github.com/Plachtaa/seed-vc.git
    uv venv -p 3.10 .venv
    VIRTUAL_ENV=.venv uv pip install torch==2.4.0 torchaudio==2.4.0 scipy==1.13.1 \
        librosa==0.10.2 "huggingface-hub>=0.28.1,<1" munch==4.0.0 einops==0.8.0 \
        descript-audio-codec==1.0.0 transformers==4.46.3 soundfile==0.12.1 \
        numpy==1.26.4 pyyaml hydra-core==1.3.2 accelerate pyworld "setuptools<80"
    # Seed-VC picks MPS on Apple Silicon and crashes on float64; force CPU:
    sed -i '' 's/torch.device("mps")/torch.device("cpu")/' seed-vc/inference.py

Run (from the seed-vc checkout):

    uv run --with soundfile python experiments/sung_replacement/prep.py samples/vampire.mp3 74.22 74.76 cases/vampire --ref 30 55
    .venv/bin/python experiments/sung_replacement/make_source.py cases/vampire sucker
    (cd seed-vc && ../.venv/bin/python inference.py --source ../cases/vampire/source.wav \
        --target ../cases/vampire/ref.wav --output ../cases/vampire/vc \
        --diffusion-steps 30 --f0-condition True --auto-f0-adjust False --fp16 False)
    .venv/bin/python experiments/sung_replacement/splice_variants.py cases/vampire clips --borrow 5.36 6.46

Results (M3, CPU): Seed-VC takes about 70-80 s per word including model load.
On vampire (sung), Whisper hears the result as "famed sucker". On squabble-up
(rapped), it hears garbled words ("get that, got my thighs").
