# Internship report — Separation of Concerns in GP Bayesian Optimization

LaTeX sources for the research internship report of **Benjamin Wilhelm**
(sciNeurotech Lab, Polytechnique Montréal; supervisor Marco Bonizzato).

## Build

```
latexmk -pdf main.tex        # runs pdflatex + bibtex automatically
# or:  pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
```

Requires a standard TeX Live (used packages: amsmath, amsthm, mathtools, bm,
geometry, natbib, hyperref, cleveref, todonotes, booktabs, subcaption, ...).
The document compiles **as is**, before you add any figure (see below).

## File map

```
main.tex                         master file (\input order = reading order)
preamble.tex                     packages, macros, theorem envs, provenance badges, figure system
references.bib                   bibliography (ALL entries verified against primary sources)
frontmatter/
  coverpage.tex                  cover page (logos, title, author, TODO dates)
  plagiarism.tex                 non-plagiarism / academic-integrity declaration
sections/
  abstract.tex
  introduction.tex               §1  (incl. contributions list + provenance legend)
  background.tex                 §2  selective, argumentative state of the art
  contrib_target_matching.tex    §3  target matching  (theory + Experiment A)
  contrib_multiobjective.tex     §4  multi-objective   (theory + Experiment C)
  contrib_prior_guided.tex       §5  prior-guided      (theory + Experiment B)
  discussion.tex                 §6
  conclusion.tex                 §7
appendices/
  appendixA_background.tex       A  math background (all imported, cited)
  appendixB_target_matching.tex  B  full proofs: kink normal form, barrier, TM regret
  appendixC_prior_guided.tex     C  full proofs: local/global, self-confirmation
  appendixD_multiobjective.tex   D  pushforward proof + Golovin–Zhang (attributed)
  appendixE_experiments.tex      E  reproducibility tables + TODO/NOT VERIFIED notes
figures/                         drop your PNGs here (Theory/ EXPA/ EXPB/ EXPC/ Logos/)
```

Section order follows the presentation: **target matching → multi-objective →
prior-guided**.

## Figures

Figures are referenced with the `\figph` / `\figauto` macros. **If a PNG is
missing, a labelled grey placeholder box is drawn showing the expected path**, so
the document always compiles. To insert the real figures, just copy the PNGs from
your `Figures_rapport/` folder into `figures/` keeping the same names, e.g.:

```
figures/Logos/logo_poly.png   figures/Logos/logo_X.png   figures/Logos/logo_lab.png
figures/Theory/Kernel_regularity.png   figures/Theory/Barriere d'approximation.png ...
figures/EXPA/gaussian_trap.png ...   figures/EXPB/... figures/EXPC/...
```

The macro auto-detects them on the next compile — no code change needed. A couple
of figure filenames still carry a browser suffix (` (1).png`, ` (3).png`) or a
space (`Barriere d'approximation.png`); either rename them or keep the names in
the `\figph{...}` calls consistent.

## Provenance colour code (shown next to every theorem)

- **established literature** — imported unchanged
- **literature, specialized here** — known result re-derived for our setting
- **personal contribution** — personal theoretical contribution
- **personal experiment** — personal experimental contribution
- **interpretation / synthesis** — interpretation proposed here
- **open question** — open / conjectural

## Things to fill in (search for `TODO` / `\TODOtext` / `\CITEQ`)

- Cover page & declaration: internship start/end dates, submission date, place.
- Experiment appendices flag every `TODO / NOT VERIFIED` item recovered from
  logs/configs (e.g. Exp C β-variant choice, incomplete `fixed_zdt2_d6` cell).
- No reference was invented: any still-missing citation is marked `\CITEQ{...}`.
