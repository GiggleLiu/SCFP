# Learning Sheets Generation Plan

This document tracks the generation of learning sheets for all SCFP topics.

## Overview

| Part | Topics | Sheets |
|------|--------|--------|
| Part I: Julia Programming | 6 | 0/6 |
| Part II: Numerical Linear Algebra | 4 | 0/4 |
| Part III: Optimization | 4 | 0/4 |
| Part IV: Simulation | 1 | 0/1 |
| Appendix | 2 | 0/2 |
| **Total** | **17** | **0/17** |

---

## Part I: Julia Programming

### 1. Terminal Environment
- **Source:** `book/chap1-julia/terminal.typ`
- **Output:** `learning-sheets/terminal-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (foundational)
- **Dependencies:** None

### 2. Version Control with Git
- **Source:** `book/chap1-julia/git.typ`
- **Output:** `learning-sheets/git-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (foundational)
- **Dependencies:** Terminal Environment

### 3. Julia Setup
- **Source:** `book/chap1-julia/julia-setup.typ`
- **Output:** `learning-sheets/julia-setup-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (foundational)
- **Dependencies:** Terminal, Git

### 4. Julia Basics
- **Source:** `book/chap1-julia/julia-basic.typ`
- **Output:** `learning-sheets/julia-basic-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (core content, largest chapter)
- **Dependencies:** Julia Setup
- **Notes:** Large chapter (1173 lines), may need to split

### 5. Package Development
- **Source:** `book/chap1-julia/julia-release.typ`
- **Output:** `learning-sheets/julia-release-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium
- **Dependencies:** Julia Basics

### 6. GPU Programming
- **Source:** `book/chap1-julia/gpu.typ`
- **Output:** `learning-sheets/gpu-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium (advanced topic)
- **Dependencies:** Julia Basics
- **Notes:** Large chapter (1149 lines)

---

## Part II: Numerical Linear Algebra

### 7. Matrix Computation
- **Source:** `book/chap2-linalg/linalg.typ`
- **Output:** `learning-sheets/linalg-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (core content)
- **Dependencies:** Julia Basics

### 8. Advanced Matrix Methods
- **Source:** `book/chap2-linalg/linalg-advanced.typ`
- **Output:** `learning-sheets/linalg-advanced-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium
- **Dependencies:** Matrix Computation
- **Notes:** Largest chapter (1314 lines)

### 9. Sparse Matrices and Graphs
- **Source:** `book/chap2-linalg/sparse.typ`
- **Output:** `learning-sheets/sparse-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium
- **Dependencies:** Matrix Computation

### 10. Tensor Networks
- **Source:** `book/chap2-linalg/tensor-network.typ`
- **Output:** `learning-sheets/tensor-network-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium (specialized topic)
- **Dependencies:** Matrix Computation, Sparse Matrices

---

## Part III: Optimization

### 11. Simulated Annealing
- **Source:** `book/chap3-optimization/simulated-annealing.typ`
- **Output:** `learning-sheets/simulated-annealing-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium
- **Dependencies:** Julia Basics

### 12. Mathematical Optimization
- **Source:** `book/chap3-optimization/linear_integer.typ`
- **Output:** `learning-sheets/linear_integer-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Medium
- **Dependencies:** Matrix Computation

### 13. Gradient-Based Optimization
- **Source:** `book/chap3-optimization/gradient-optimization.typ`
- **Output:** `learning-sheets/gradient-optimization-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (connects to AD)
- **Dependencies:** Matrix Computation

### 14. Automatic Differentiation
- **Source:** `book/chap3-optimization/ad.typ`
- **Output:** `learning-sheets/ad-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (modern technique)
- **Dependencies:** Gradient-Based Optimization
- **Notes:** Large chapter (1032 lines)

---

## Part IV: Simulation

### 15. Monte Carlo Methods
- **Source:** `book/chap4-simulation/MCMC.typ`
- **Output:** `learning-sheets/MCMC-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** High (capstone topic)
- **Dependencies:** Simulated Annealing, Julia Basics

---

## Appendix

### 16. Plotting with CairoMakie
- **Source:** `book/appendix/plotting.typ`
- **Output:** `learning-sheets/plotting-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Low (reference material)
- **Dependencies:** Julia Basics

### 17. Compressed Sensing
- **Source:** `book/chap2-linalg/compressed.typ`
- **Output:** `learning-sheets/compressed-learning-sheet.typ`
- **Status:** ⬜ Not started
- **Priority:** Low (specialized topic)
- **Dependencies:** Matrix Computation, Mathematical Optimization

---

## Generation Order (Recommended)

Based on dependencies and priority:

**Phase 1: Foundations (6 sheets)**
1. ⬜ `terminal` - No dependencies
2. ⬜ `git` - Depends on terminal
3. ⬜ `julia-setup` - Depends on terminal, git
4. ⬜ `julia-basic` - Core content
5. ⬜ `linalg` - Core content
6. ⬜ `gradient-optimization` - Needed for AD

**Phase 2: Core Topics (5 sheets)**
7. ⬜ `ad` - High priority, modern
8. ⬜ `MCMC` - Capstone simulation
9. ⬜ `simulated-annealing` - Optimization
10. ⬜ `linear_integer` - Optimization
11. ⬜ `sparse` - Linear algebra

**Phase 3: Advanced Topics (4 sheets)**
12. ⬜ `linalg-advanced` - Advanced LA
13. ⬜ `tensor-network` - Specialized
14. ⬜ `julia-release` - Package dev
15. ⬜ `gpu` - Advanced computing

**Phase 4: Appendix (2 sheets)**
16. ⬜ `plotting` - Reference
17. ⬜ `compressed` - Specialized

---

## Commands

Generate a single learning sheet:
```
generate learning sheet for <topic_name>
```

Example:
```
generate learning sheet for julia-basic
```

---

## Status Legend

- ⬜ Not started
- 🔄 In progress
- ✅ Completed
- ⚠️ Needs revision
- ❌ Blocked

---

## Notes

- Large chapters (>1000 lines) may need extra review time
- Some topics have slide files that can provide additional context
- Prioritize foundational topics first to establish patterns
- Review first few sheets carefully to establish quality baseline
