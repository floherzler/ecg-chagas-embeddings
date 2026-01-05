## Thesis (LaTeX)

This folder contains the LaTeX sources for the master thesis.

### VS Code
- Install the recommended extension: **LaTeX Workshop**.
- Open `thesis/main.tex` and use LaTeX Workshop “Build LaTeX project”.

### Build (CLI)
From the repo root:

```bash
latexmk -cd -r thesis/latexmkrc -pdf -interaction=nonstopmode -halt-on-error thesis/main.tex
```

Outputs go to `thesis/.out/`.

