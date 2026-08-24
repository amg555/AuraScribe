-- Optional spectral noise reduction (denoise.rs): before transcription, subtract the background
-- noise spectrum estimated from the pauses, so steady noise (fans, AC, road hum) is reduced while
-- the voice is kept. OFF by default (0) -- it is safe on clean audio, but the real-world benefit is
-- mic- and environment-dependent, so the user opts in from Settings. Existing installs get 0.
ALTER TABLE settings ADD COLUMN noise_suppression INTEGER NOT NULL DEFAULT 0;
