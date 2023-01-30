# AFBA_Applications

This repository contains the algorithms and applications developed in Reyes, F. Rafael, "Speeding up Asymmetric Forward-Backward-Adjoint
Splitting Algorithms: Methods and Applications" from the Mathematical Engineering Master's Thesis - KU LEUVEN.

The repository `ProximalAlgorithms.jl` is cloned within this repository, for its original implementation go to https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl.git.

The algorithms Adaptive AFBA (`AAFBA.jl`) [1], Generalized Primal Dual Algorithm with Linesearch (`GPDAL.jl`) [2], Golden Ratio Primal-Dual Algorithm with Linesearch (`GRPDA.jl`) [3],
and an Extended Golden Ratio Primal-Dual Algorithm with Linesearch (`EGRPDA.jl`) are implemented within the cloned repository.

Linear mappings not available in the `ProximalOperators.jl` package were also developed; these are a 2D Total Variation operator using Sparse Matrices, and a Hankel Matrix subsetter.
The proximal function of the quadratic perturbation and a row by row implementation of the L2 Indicator Ball were also developed. These can be found in the respective application where they were used.

# Applications
## Total Variation l1 Denoising
The implementation of TV-l1 denoising is shown in `TV_Denoising.jl`, inside the TV_Denoising folder.

`TV_Denoising_all` and `TV_Denoising_precon` are scripts to generate the plots shown in the master's thesis.

## Rank Minimization via regularized Spectral Norm minimization
The implementation of Rank Minimization is shown in `Rank_Min.jl`, inside the Rank_Min folder.

`Rank_Min_all_1.jl`, `Rank_Min_all_2.jl` and `Rank_Min_all_precon.jl` are scripts to generate the plots shown in the master's thesis.

# References

[1] P. Latafat et al. “Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient”. In: arXiv (2023). doi: https://doi.org/10.48550/arXiv.2301.04431.

[2] Yura Malitsky and Thomas Pock. “A First-Order Primal-Dual Algorithm with Linesearch”. In: SIAM Journal on Optimization 28 (2018), pp. 411–432. doi: https://doi.org/10.1137/16M1092015.

[3] X.K. Chang, J. Yang, and H. Zhang. “Golden Ratio Primal-Dual Algorithm with Linesearch”. In: SIAM Journal on Optimization 32 (2022), pp. 1584–1613. doi: https://doi.org/10.1137/21M1420319.

[4] P. Latafat and P. Patrinos. “Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators”. In: Comput Optim Appl 68 (2017), pp. 57–93. doi: https://doi.org/10.1007/s10589-017-9909-6.
