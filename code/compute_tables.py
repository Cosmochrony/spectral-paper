"""
compute_tables.py
=================
Reproduces all numerical tables in SpectralRelaxation 1.0 and prints
the full diagnostic output used in the paper.

Tables produced:
  Table 1 -- Reading A: c_i(p) and mass ratios for 2T ord-3
  Table 2 -- Reading A: c_i(p) and mass ratios for 2I ord-4
  Table 3 -- Reading A: c_i(p) and mass ratios for 2I ord-5
  Table 4 -- Reading B: log10(M_i) and log10 mass ratios (all cases)
  Table 5 -- O1 support table: 2I ord-5 across p = 5..499
  Table 6 -- Exact exit-p values for 2I ord-5 (analytical)
  Comparison with observed SM mass ratios

Usage:
  python compute_tables.py
"""

import numpy as np
from spectral_relaxation_lib import (
    lam_support, km_cdf, normalised_levels, levels_in_support,
    reading_A, reading_B, exit_p_exact, support_table,
    ADE_CASES, SM_RATIOS
)

SEP  = "=" * 72
SEP2 = "-" * 72

# Prime values considered
P_ALL = [5, 13, 29, 53, 101, 199, 499]


def hdr(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def reading_A_table(case_key, p_values):
    case  = ADE_CASES[case_key]
    dims  = case["dims"]
    norms = normalised_levels(case_key)

    hdr(f"TABLE -- Reading A: {case['label']}")
    print(f"Normalised levels: {[f'{v:.4f}' for v in norms]}")
    print(f"Rep dimensions:    {dims}")
    print()
    header = (f"{'p':>5}  {'lam_-':>7}  {'lam_+':>7}  "
              f"{'c1':>7}  {'c2':>7}  {'c3':>7}  "
              f"{'M1/M2':>8}  {'M2/M3':>8}  {'M1/M3':>10}")
    print(header)
    print(SEP2)

    for p in p_values:
        lm, lp = lam_support(p)
        res = reading_A(case_key, p)
        if res is None:
            ok, flags = levels_in_support(case_key, p)
            flag_str = [("IN" if f else "OUT") for f in flags]
            print(f"{p:5d}  [{lm:.4f},{lp:.4f}]  "
                  f"levels: {flag_str}  -- skipped")
        else:
            c = res["c"]
            print(f"{p:5d}  {lm:7.4f}  {lp:7.4f}  "
                  f"{c[0]:7.4f}  {c[1]:7.4f}  {c[2]:7.4f}  "
                  f"{res['r12']:8.4f}  {res['r23']:8.4f}  "
                  f"{res['r13']:10.6f}")


def reading_B_table(p_values):
    hdr("TABLE -- Reading B: log10(M_i) = dim_i * log10(c_i)")
    cases_to_show = ["2T_ord3", "2I_ord4", "2I_ord5"]

    for case_key in cases_to_show:
        case  = ADE_CASES[case_key]
        dims  = case["dims"]
        norms = normalised_levels(case_key)
        print(f"\n  {case['label']}")
        print(f"  Levels: {[f'{v:.4f}' for v in norms]},  dims: {dims}")
        header = (f"  {'p':>5}  {'c1':>7}  {'c2':>7}  {'c3':>7}  "
                  f"{'log10M1':>8}  {'log10M2':>8}  {'log10M3':>8}  "
                  f"{'lg r12':>7}  {'lg r23':>7}  {'lg r13':>7}  order")
        print(header)
        print("  " + "-" * 68)
        for p in p_values:
            res = reading_B(case_key, p)
            if res is None:
                lm, lp = lam_support(p)
                print(f"  {p:5d}  [{lm:.4f},{lp:.4f}]  -- outside support")
                continue
            c = res["c"]
            lm = res["log10_M"]
            order_str = "".join(str(i+1) for i in res["mass_order"])
            print(f"  {p:5d}  {c[0]:7.4f}  {c[1]:7.4f}  {c[2]:7.4f}  "
                  f"{lm[0]:8.1f}  {lm[1]:8.1f}  {lm[2]:8.1f}  "
                  f"{res['log10_r12']:7.1f}  {res['log10_r23']:7.1f}  "
                  f"{res['log10_r13']:7.1f}  {order_str}")


def o1_support_table():
    case_key = "2I_ord5"
    case     = ADE_CASES[case_key]
    norms    = normalised_levels(case_key)

    hdr(f"TABLE -- O1: Support narrowing for {case['label']}")
    print(f"Levels: lambda1={norms[0]:.4f}, lambda2={norms[1]:.4f}, "
          f"lambda3={norms[2]:.4f}")
    print()
    header = (f"{'p':>5}  {'lam_-':>7}  {'lam_+':>7}  "
              f"{'l1 in?':>7}  {'l2 in?':>7}  {'l3 in?':>7}  "
              f"{'c1':>8}  {'c2':>8}  {'c3':>8}")
    print(header)
    print(SEP2)

    rows = support_table(case_key, P_ALL)
    for row in rows:
        p   = row["p"]
        lm  = row["lam_minus"]
        lp  = row["lam_plus"]
        ins = ["  yes" if f else "   no" for f in row["in_support"]]
        c   = [f"{v:8.4f}" if v is not None else "     ---" for v in row["c"]]
        print(f"{p:5d}  {lm:7.4f}  {lp:7.4f}  "
              f"{ins[0]}  {ins[1]}  {ins[2]}  "
              f"{c[0]}  {c[1]}  {c[2]}")


def exit_p_table():
    case_key = "2I_ord5"
    norms    = normalised_levels(case_key)
    names    = ["lambda_1 = 20/24", "lambda_2 = 24/24", "lambda_3 = 30/24"]

    hdr("TABLE -- O1: Exact analytical exit-p values for 2I ord-5")
    print(f"{'Level':20s}  {'Exact sqrt(p_exit)':35s}  {'p_exit':>10}")
    print(SEP2)

    for name, lv in zip(names, norms):
        p_exit, desc = exit_p_exact(lv)
        if p_exit is None:
            print(f"{name:20s}  {'never exits':35s}  {'---':>10}")
        else:
            print(f"{name:20s}  {desc[:35]:35s}  {p_exit:10.4f}")

    print()
    print("Interpretation:")
    print("  lambda_3 exits first (p_exit ~ 62): mode 3 stabilises at smallest n")
    print("  => M_3 is the heaviest generation under O1 scenario")
    print("  lambda_1 exits last  (p_exit ~142): mode 1 stabilises last")
    print("  => M_1 is the lightest generation")
    print("  lambda_2 = 1 never exits: mode 2 stabilises when modes 1,3 have")
    print("  already stabilised -- it is the intermediate generation")
    print("  Mass ordering: M_3 > M_2 > M_1 (restores SpectralStratigraphy order)")


def sm_comparison():
    hdr("COMPARISON WITH STANDARD MODEL MASS RATIOS")
    print(f"{'Sector':20s}  {'r12':>10}  {'r23':>10}  {'r13':>12}")
    print(SEP2)
    for key, info in SM_RATIOS.items():
        print(f"{info['label']:20s}  {info['r12']:10.2e}  "
              f"{info['r23']:10.2e}  {info['r13']:12.2e}")

    print()
    print("Reading A predictions (best case, 2I ord-5, p=53):")
    res = reading_A("2I_ord5", 53)
    if res:
        print(f"  M1/M2 = {res['r12']:.4f},  M2/M3 = {res['r23']:.4f},  "
              f"M1/M3 = {res['r13']:.4f}")
        print(f"  Gap vs leptons M1/M3: factor ~{res['r13']/SM_RATIOS['charged_leptons']['r13']:.0f}")

    print()
    print("Reading B predictions (2I ord-5, p=29):")
    res = reading_B("2I_ord5", 29)
    if res:
        print(f"  log10(M1/M2) = {res['log10_r12']:.1f},  "
              f"log10(M2/M3) = {res['log10_r23']:.1f},  "
              f"log10(M1/M3) = {res['log10_r13']:.1f}")
        print(f"  Comparison with log10(m_e/m_tau) = {np.log10(SM_RATIOS['charged_leptons']['r13']):.1f}")
        print(f"  Comparison with log10(m_u/m_t)   = {np.log10(SM_RATIOS['up_quarks']['r13']):.1f}")


def c2_identity_check():
    hdr("STRUCTURAL IDENTITY: F_KM(1) = 1/2 for all p")
    print("Exact result from symmetry rho_KM(lambda) = rho_KM(2-lambda).")
    print()
    print(f"{'p':>5}  {'F_KM(1)':>12}  {'error from 1/2':>15}")
    print(SEP2)
    for p in [5, 13, 29, 53, 101, 199]:
        val = km_cdf(1.0, p)
        print(f"{p:5d}  {val:12.10f}  {abs(val-0.5):15.2e}")


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    print(SEP)
    print("SpectralRelaxation 1.0 -- Numerical Tables")
    print("Cosmochrony / Bounded Admissibility Theory")
    print(SEP)

    # Tables 1-3: Reading A
    for case_key in ["2T_ord3", "2I_ord4", "2I_ord5"]:
        reading_A_table(case_key, P_ALL)

    # Table 4: Reading B
    reading_B_table(P_ALL)

    # Table 5: O1 support narrowing
    o1_support_table()

    # Table 6: Exact exit-p values
    exit_p_table()

    # Structural identity c2 = 1/2
    c2_identity_check()

    # SM comparison
    sm_comparison()

    print(f"\n{SEP}")
    print("All tables complete.")
    print(SEP)
