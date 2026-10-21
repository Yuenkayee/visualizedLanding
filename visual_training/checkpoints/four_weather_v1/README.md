# Default four-weather inference checkpoint

`best.pt` is included in Git so a fresh clone can run `main/navigation_observer.yaml`
without separately transferring training data or weights. It is a regular Git
binary, not a Git LFS pointer.

- Scenarios: clear, fog, night, night_fog.
- File size: 1,977,791 bytes.
- SHA256: `1605b4a258414d931f08c630dc1bd83c3946f1a91bfa595614ee0e54df186ab3`.
- This is the existing local checkpoint; it was not retrained during deployment work.
- Other checkpoints and training logs remain ignored. Use `--checkpoint` to
  evaluate another training result.
