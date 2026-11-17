# Figures_rapport — carte des figures

Dossier d'images (PNG uniquement, 37 fichiers, ~12 Mo) servant d'illustrations au rapport de
stage sur l'optimisation bayésienne par processus gaussiens (GPBO) et le cadre *SoC*
(Set-of-Constraints / scalarisation). **Aucun code, aucune donnée** ici : uniquement les sorties
graphiques finales. Les scripts qui les produisent sont ailleurs (`D:\Stage_recherche_soc\Code`).

```
Figures_rapport/
├── Theory/   (4)   fondements théoriques : régularité RKHS, postérieur GP
├── EXPA/     (11)  expérience A — modéliser f vs modéliser |f − T|
├── EXPB/     (13)  expérience B — injection d'un prior π dans l'acquisition
├── EXPC/     (6)   expérience C — multi-objectif : Pareto, hypervolume, scalarisation
└── Logos/    (3)   logos institutionnels pour la page de garde
```

## Conventions transversales

- **Préfixe numérique** (`01_`, `02_`, `03_`, `04_`) = étape du pipeline d'analyse, pas ordre dans
  le rapport : `01` = setup/benchmark, `02` = trajectoires postérieur+acquisition, `03` = métriques
  agrégées sur seeds, `04` = diagnostics de méthode. Les fichiers **sans préfixe** sont des variantes
  ponctuelles régénérées à la main.
- **Suffixe = benchmark** : `gaussian_trap`, `branin`, `hartmann`, `double_well`, complété par
  `_good_prior` / `_bad_prior` en EXPB.
- **Code couleur EXPA** : bleu = `RIGHT` (GP sur f), rouge = `WRONG` (GP sur |f − T|).
- **Code couleur EXPB** : gris = `vanilla UCB`, bleu = `απ-UCB (SoC)`, rose = `λπ fixed (std)`,
  vert = `λπ adaptive (std)`.
- **Boîtes à moustaches** : boîte 25–75 %, moustaches 5–95 %, trait plein = médiane, tirets = moyenne.
  Les courbes agrégées sont des médianes sur 100 seeds (EXPA) ou ~50 seeds (EXPB) avec bande
  bootstrap 95 %.

## Theory/ — objets théoriques

| Fichier | Contenu |
|---|---|
| `Kernel_regularity.png` | 3 panneaux de trajectoires GP échantillonnées : Matérn-1/2 (rugueux), Matérn-5/2 (moyennement lisse), RBF (C∞). Illustre l'effet de ν sur la régularité. |
| `Barriere d'approximation.png` | Borne inférieure ε∞(B) ≳ B^(−1/(ν−1)) en log-log, budget RKHS B de 1 à 10⁴, une courbe par ν ∈ {1.25, 1.5, 2, 3, 5, 10}. Montre que la barrière s'effondre quand ν → 1. |
| `posterior_mean_band.png` | Vignette : postérieur GP (moyenne + bande ±2σ) contre vérité terrain pointillée à t = 5, avec seuil cible en vert. |
| `posterior_aq.png` | Vignette dépouillée (sans titre ni légende) : deux fonctions d'acquisition bleue/rouge avec leurs argmax marqués. Sert d'encart. |

## EXPA/ — modéliser f (RIGHT) vs modéliser |f − T| (WRONG)

Question : pour atteindre une **cible** T, faut-il mettre le GP sur la fonction brute f ou sur
l'écart transformé |f − T| ? Conclusion des figures : la transformation en valeur absolue casse la
régularité (point de rebroussement au croisement de la cible) et dégrade fortement la performance.

**Mise en place du contre-exemple 1D (« gaussian trap ») :**

| Fichier | Contenu |
|---|---|
| `gaussian_trap.png` | f(x) brute, seuil T = 0.38, croisement de la cible marqué. C'est ce que modélise RIGHT. |
| `gaussian_trap_abs.png` | La même chose transformée en \|f(x) − T\| : le kink en x ≈ 0.62 est visible. C'est ce que modélise WRONG. |
| `gaussian_trap_failure.png` | Colonne unique à t = 6 (postérieur RIGHT / postérieur WRONG / acquisitions superposées) : instantané de l'échec de WRONG. |

**Trajectoires :**

| Fichier | Contenu |
|---|---|
| `02b_posterior_and_acquisition_early_evolution.png` | Grille 3×10, t = 1…10 (seed 1). Ligne 1 postérieur RIGHT, ligne 2 postérieur WRONG, ligne 3 acquisitions des deux. RIGHT converge vers la vérité dès t ≈ 5 ; WRONG garde une incertitude énorme et une acquisition multi-modale. |
| `02_acquisition_evolution_2d.png` | Grille 2×6 en 2D (init, t = 5, 10, 20, 40, 50), heatmap de α(x) + points visités reliés. Ligne haute RIGHT : les requêtes se placent proprement sur l'ellipse de niveau cible. Ligne basse WRONG : dispersion et amas hors-cible. |

**Métriques agrégées (100 seeds) :**

| Fichier | Contenu |
|---|---|
| `03_cumulative_regret_gaussian_trap.png` | 2 panneaux (regret instantané cumulé / best-so-far cumulé). RIGHT plafonne, WRONG croît linéairement — le résultat le plus net du lot. |
| `03_cumulative_regret_branin.png` | Même figure sur Branin (2D). Écart moins spectaculaire mais de même sens ; le panneau best-so-far y est quasi confondu. |
| `03_failure_rate_gaussian_trap.png` | Barplot du taux d'échec (seuil 0.015) : RIGHT 0.05 vs WRONG 0.87. |
| `03_final_distribution_gaussian_trap.png` | Boxplots des deux regrets cumulés finaux. |
| `03_final_distribution_branin.png` | Idem sur Branin (échelle log, ×~1.8 en faveur de RIGHT sur le regret instantané). |
| `03_final_distribution_hartmann.png` | Idem sur Hartmann (6D) : l'écart se réduit encore, panneau best-so-far non concluant. |

## EXPB/ — prior π injecté dans l'acquisition

Quatre méthodes comparées : `vanilla UCB` (sans prior), `απ-UCB (SoC)`, `λπ fixed (std)`,
`λπ adaptive (std)`. Benchmark principal : un **double puits 1D** (optimum global en x ≈ 0.70,
optimum local en x ≈ 0.25), décliné en *bon prior* (π centré sur l'optimum global) et *mauvais
prior* (π centré sur le local).

| Fichier | Contenu |
|---|---|
| `benchmark_double_well_good.png` | Double puits + prior π **bien** placé (centré sur x★ ≈ 0.70). |
| `01_benchmark_and_prior.png` | Même figure, prior **mal** placé (centré sur le local x ≈ 0.25). C'est le cas adverse de référence. |
| `03_posterior_aq_double_well_bad (3).png` | Grille 2×7 (t = 1, 2, 5…9) pour `λπ adaptive` seul, mauvais prior : le postérieur reste verrouillé sur le mode local, le pic global (en gris) n'est jamais exploré. Illustration visuelle de l'échec. |
| `03_best_so_far_regret.png` | Regret best-so-far (log), double puits mauvais prior : vanilla et SoC descendent à 10⁻⁷, les deux `λπ` restent bloqués à ~0.6. |
| `03_best_so_far_regret_branin_bad_prior.png` | Idem Branin : tout le monde converge, `λπ fixed` seulement retardé (~15 itérations de pénalité). |
| `03_best_so_far_regret_hartmann_bad_prior.png` | Idem Hartmann : `λπ fixed` ne converge jamais dans le budget, `λπ adaptive` converge très tard (t ≈ 90). |
| `03_final_regret_distributions.png` | Boxplots regret final best-so-far / cumulé, double puits mauvais prior. Distribution bimodale pour les `λπ` (soit ils trouvent, soit pas du tout). |
| `03_final_regret_distributions_double_well_bad (1).png` | ⚠️ **Doublon octet-pour-octet** du précédent (même MD5). Un des deux peut être supprimé. |
| `03_final_regret_distributions_branin_bad_prior.png` | Idem Branin : regrets finaux tous ~10⁻⁴, la différence est dans le regret cumulé (`λπ fixed` ≈ 300 vs vanilla ≈ 190). |
| `03_final_regret_distributions_hartmann_bad_prior.png` | Idem Hartmann : `λπ fixed` clairement dégradé sur les deux panneaux. |
| `03_hitting_time_failure_double_well_good.png` | Temps d'atteinte τ (ε = 0.35) + taux d'échec, **bon** prior : 0 % d'échec sauf `λπ adaptive` à 36 %. |
| `hitting_time_double_well_bad.png` | Idem, **mauvais** prior : `λπ fixed` 50 %, `λπ adaptive` 64 %, vanilla et SoC 0 %. |
| `04_lambda_trajectory.png` | Diagnostic : trajectoire du coefficient adaptatif λ̂_t. Médiane ≈ 0.79 (le prior reste largement accepté), moyenne ≈ 0.40, forte variance inter-seeds — explique pourquoi `λπ adaptive` échoue par intermittence. |

## EXPC/ — multi-objectif

| Fichier | Contenu |
|---|---|
| `Pareto_front_example.png` | Schéma pédagogique (image importée, style externe) : front de Pareto, solutions dominées vs non-dominées, comparaison A/B/C. |
| `Hypervolume_example.png` | Schéma pédagogique : hypervolume comme aire dominée jusqu'au point de référence. |
| `Pareto_front_right.png` | Résultat 2 panneaux (seed 1), espace objectif : `right_algorithm_2` (gauche) vs `right_algorithm_1` (droite). Le second couvre le front vrai bien plus densément. |
| `Pareto_front_wrong_alg2.png` | Même vue pour `wrong_algorithm_2` : points agglutinés sur les bords f₁ = 0 et f₁ = 1, front mal couvert au centre. |
| `hypervolume.png` | Hypervolume exact vs nombre d'évaluations, 3 algos + HV\* vrai en pointillé. **Attention : le label de l'axe x est rogné** dans le PNG. |
| `Scalarization_Multiobjectif.png` | Grille 2×3 : scalarisation fixe linéaire à 80° (haut) vs hypervolume à 10° (bas), colonnes best-so-far / regret simple / regret cumulé, RIGHT (SoC) vs WRONG. À 80° les deux sont indiscernables ; à 10° SoC domine nettement. |

## Logos/

`logo_poly.png` (Polytechnique Montréal), `logo_X.png` (École polytechnique / IP Paris),
`logo_lab.png` (sciNeurotech Lab — Marco Bonizzato). Page de garde uniquement.

## Points d'attention

- Deux fichiers portent encore un suffixe de téléchargement navigateur : `... (1).png` et `... (3).png`.
- `03_final_regret_distributions.png` et `03_final_regret_distributions_double_well_bad (1).png` sont
  strictement identiques.
- `03_final_regret_distributions.png` et `03_best_so_far_regret.png` n'ont pas de suffixe de benchmark
  mais correspondent bien au **double puits / mauvais prior**.
- Résolutions très hétérogènes (de ~450 px à ~6450 px de large) : vérifier le rendu avant insertion LaTeX.
