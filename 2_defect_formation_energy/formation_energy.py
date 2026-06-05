#!/usr/bin/env python3
"""
========================================================================
 Defect formation energy diagram for Si:S_Si and Si:V_Si
========================================================================

Outputs:
  1) chempot_diagram.png
  2) Ef_all_charge_S_poor.png
  3) Ef_all_charge_S_rich.png

Included charge states:
  S_Si : q = 0, +1, +2
  V_Si : q = 0, -2

Chemical potential limits:
  S-rich:
      mu_S = mu_S_ref
      Delta mu_S = 0

  S-poor:
      mu_Si = mu_Si_bulk
      mu_Si_bulk + 2 mu_S = E(SiS2)
      Delta mu_S = (E(SiS2) - mu_Si_bulk)/2 - mu_S_ref

Notes:
  - Raw QE total energies are used.
  - QE "Total+Makov-Payne energy" lines are ignored.
  - No finite-size charge correction is applied.
========================================================================
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY2EV = 13.605693122994

# ============================================================
# User settings
# ============================================================
BASE = "."

PATHS = {
    "bulk_Si":  f"{BASE}/1_Si/si.vc-relax.out",
    "SiS2":     f"{BASE}/2_SiS2/sis2.vc-relax.out",
    "S2":       f"{BASE}/3_S2/s2.relax.out",
    "perfect":  f"{BASE}/4_Si_bulk_supercell/si_bulk.scf.out",

    "S_Si_q0":  f"{BASE}/5_S_Si_substitution/charge_0/s_si.relax.out",
    "S_Si_q1":  f"{BASE}/5_S_Si_substitution/charge_1/s_si.relax.out",
    "S_Si_q2":  f"{BASE}/5_S_Si_substitution/charge_2/s_si.relax.out",

    "V_Si_q0":  f"{BASE}/6_V_Si_vacancy/charge_0/v_si.relax.out",
    "V_Si_qm2": f"{BASE}/6_V_Si_vacancy/charge_-2/v_si.relax.out",
}

N_SI_BULK_ATOMS = 8
N_SIS2_FU = 4
N_S2_ATOMS = 2

BAND_GAP = 0.60   # eV, PBE Si gap


# ============================================================
# Parsers
# ============================================================
def read_total_energy(path):
    """
    Read last raw QE total energy in Ry.
    Ignore 'Total+Makov-Payne energy'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    e = None
    with open(path) as f:
        for line in f:
            t = line.strip()
            if t.startswith("!") and "Makov" not in t and "total energy" in t:
                m = re.search(r"=\s*(-?\d+\.\d+)", t)
                if m:
                    e = float(m.group(1))

    if e is None:
        raise ValueError(f"No raw '! total energy' found in {path}")

    return e


def read_vbm(path):
    """
    Read VBM from perfect supercell output.
    If 'highest occupied' is not found, use Fermi energy.
    """
    with open(path) as f:
        txt = f.read()

    m = re.search(r"highest occupied.*?:\s*(-?\d+\.\d+)", txt)
    if m:
        return float(m.group(1))

    m = re.search(r"Fermi energy is\s*(-?\d+\.\d+)", txt)
    if m:
        return float(m.group(1))

    raise ValueError(f"Could not find VBM/Fermi energy in {path}")


def charge_label(q):
    if q > 0:
        return f"+{q}"
    if q < 0:
        return f"{q}"
    return "0"


# ============================================================
# Reference energies and chemical potentials
# ============================================================
E_bulk_Si = read_total_energy(PATHS["bulk_Si"]) * RY2EV
mu_Si = E_bulk_Si / N_SI_BULK_ATOMS

E_SiS2_fu = read_total_energy(PATHS["SiS2"]) * RY2EV / N_SIS2_FU
mu_S_ref = read_total_energy(PATHS["S2"]) * RY2EV / N_S2_ATOMS

# S-rich
dmu_S_rich = 0.0

# S-poor from SiS2 equilibrium under Si-rich condition:
# mu_Si_bulk + 2 mu_S = E(SiS2)
dmu_S_poor = (E_SiS2_fu - mu_Si) / 2.0 - mu_S_ref

mu_S_at = {
    "S-poor": mu_S_ref + dmu_S_poor,
    "S-rich": mu_S_ref + dmu_S_rich,
}

E_perfect = read_total_energy(PATHS["perfect"]) * RY2EV
E_VBM = read_vbm(PATHS["perfect"])

E = {
    "S_Si_q0":  read_total_energy(PATHS["S_Si_q0"]) * RY2EV,
    "S_Si_q1":  read_total_energy(PATHS["S_Si_q1"]) * RY2EV,
    "S_Si_q2":  read_total_energy(PATHS["S_Si_q2"]) * RY2EV,
    "V_Si_q0":  read_total_energy(PATHS["V_Si_q0"]) * RY2EV,
    "V_Si_qm2": read_total_energy(PATHS["V_Si_qm2"]) * RY2EV,
}

print("Reference values")
print(f"  mu_Si                 = {mu_Si:.6f} eV/atom")
print(f"  E_SiS2_fu             = {E_SiS2_fu:.6f} eV/f.u.")
print(f"  mu_S_ref              = {mu_S_ref:.6f} eV/atom")
print(f"  Delta mu_S poor       = {dmu_S_poor:.6f} eV")
print(f"  Delta mu_S rich       = {dmu_S_rich:.6f} eV")
print(f"  E_perfect             = {E_perfect:.6f} eV")
print(f"  E_VBM                 = {E_VBM:.6f} eV")
print()

print("Defect total energies")
for k, v in E.items():
    print(f"  {k:10s} = {v:.6f} eV")
print()


# ============================================================
# Chemical potential diagram
# ============================================================
def plot_chempot_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.6))

    ax.hlines(0, dmu_S_poor, dmu_S_rich, color="tab:green", lw=4)
    ax.plot(dmu_S_rich, 0, "o", color="tab:green", ms=11)
    ax.plot(dmu_S_poor, 0, "o", color="tab:orange", ms=11)

    ax.annotate(
        "S-rich limit\n"
        "$\\mu_S=\\mu_S^{ref}$\n"
        "$\\Delta\\mu_S=0$",
        (dmu_S_rich, 0),
        textcoords="offset points",
        xytext=(-12, 22),
        ha="right",
        va="bottom",
        color="tab:green",
        fontsize=10,
    )

    ax.annotate(
        "S-poor limit\n"
        "SiS$_2$ equilibrium\n"
        "$\\mu_{Si}^{bulk}+2\\mu_S=E(SiS_2)$\n"
        f"$\\Delta\\mu_S={dmu_S_poor:.2f}$ eV",
        (dmu_S_poor, 0),
        textcoords="offset points",
        xytext=(12, -48),
        ha="left",
        va="top",
        color="tab:orange",
        fontsize=10,
    )

    formula_text = (
        "Chemical potential constraints:\n"
        "$\\Delta\\mu_S=\\mu_S-\\mu_S^{ref}$\n"
        "S-rich: $\\mu_S=\\mu_S^{ref}$\n"
        "S-poor: SiS$_2$ boundary with $\\mu_{Si}=\\mu_{Si}^{bulk}$"
    )

    ax.text(
        0.5,
        0.88,
        formula_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85),
    )

    pad = 0.15 * abs(dmu_S_poor) if dmu_S_poor != 0 else 0.2
    ax.set_xlim(dmu_S_poor - pad, dmu_S_rich + pad)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("$\\Delta\\mu_S$ (eV)")
    ax.set_title("Allowed S chemical potential range")

    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()

    fname = "chempot_diagram.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"Saved: {os.path.abspath(fname)}")


# ============================================================
# Formation energy model
# ============================================================
DEFECTS = {
    "S_Si": {
        "states": [
            (0, "S_Si_q0"),
            (1, "S_Si_q1"),
            (2, "S_Si_q2"),
        ],
        "color": "tab:red",
        "label": "S$_{Si}$",
        "chem": lambda mu_S: mu_Si - mu_S,
    },
    "V_Si": {
        "states": [
            (0, "V_Si_q0"),
            (-2, "V_Si_qm2"),
        ],
        "color": "tab:blue",
        "label": "V$_{Si}$",
        "chem": lambda mu_S: mu_Si,
    },
}


def Ef(key, chem, q, EF):
    """
    Formation energy at Fermi level EF.
    EF is measured from VBM.
    """
    return E[key] - E_perfect + chem + q * (E_VBM + EF)


def transition_levels(states):
    """
    Charge transition levels between adjacent charge states.
    Chemical potential cancels for charge states of the same defect.
    """
    s = sorted(states, key=lambda x: -x[0])
    out = []

    for i in range(len(s) - 1):
        q1, k1 = s[i]
        q2, k2 = s[i + 1]

        if q1 == q2:
            continue

        a1 = E[k1] + q1 * E_VBM
        a2 = E[k2] + q2 * E_VBM

        EF_t = (a2 - a1) / (q1 - q2)
        out.append((EF_t, q1, q2, k1))

    return out


# ============================================================
# Formation energy plots
# ============================================================
def plot_formation_energy():
    EF = np.linspace(0.0, BAND_GAP, 400)

    for limit in ("S-poor", "S-rich"):
        mu_S = mu_S_at[limit]

        plt.figure(figsize=(7.2, 5.6))

        for name, d in DEFECTS.items():
            chem = d["chem"](mu_S)

            lines = []
            for q, key in d["states"]:
                y = np.array([Ef(key, chem, q, ef) for ef in EF])
                lines.append(y)

                plt.plot(
                    EF,
                    y,
                    color=d["color"],
                    lw=0.9,
                    ls="--",
                    alpha=0.45,
                )

                x_text = BAND_GAP * 0.92
                y_text = Ef(key, chem, q, x_text)

                plt.text(
                    x_text,
                    y_text,
                    f"q={charge_label(q)}",
                    color=d["color"],
                    fontsize=8,
                    alpha=0.75,
                    va="center",
                )

            lines = np.array(lines)
            envelope = lines.min(axis=0)

            plt.plot(
                EF,
                envelope,
                color=d["color"],
                lw=2.5,
                label=d["label"],
            )

            for EF_t, q1, q2, k1 in transition_levels(d["states"]):
                tag = "in gap" if 0 <= EF_t <= BAND_GAP else "outside gap"

                print(
                    f"{limit:7s} {name:5s} "
                    f"eps({charge_label(q1)}/{charge_label(q2)}) "
                    f"= {EF_t:.4f} eV above VBM  ({tag})"
                )

                if 0 <= EF_t <= BAND_GAP:
                    y_t = Ef(k1, chem, q1, EF_t)

                    plt.plot(EF_t, y_t, "ko", ms=5, zorder=5)

                    plt.annotate(
                        f"$\\varepsilon$({charge_label(q1)}/{charge_label(q2)})={EF_t:.2f}",
                        (EF_t, y_t),
                        textcoords="offset points",
                        xytext=(5, 6),
                        fontsize=8,
                    )

        plt.axvline(0.0, color="gray", ls=":", lw=0.8)
        plt.axvline(BAND_GAP, color="gray", ls=":", lw=0.8)

        plt.xlabel("Fermi level $E_F$ (eV)   [0 = VBM]")
        plt.ylabel("Formation energy $E_f$ (eV)")
        plt.title(f"Defect formation energy in Si ({limit} limit)")
        plt.xlim(0.0, BAND_GAP)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = f"Ef_all_charge_{limit.replace('-', '_')}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

        print(f"Saved: {os.path.abspath(fname)}")


# ============================================================
# Run
# ============================================================
print("Current working directory:")
print(os.getcwd())
print()

plot_chempot_diagram()
plot_formation_energy()

print("\nDone.")
print("Generated files:")
print("  chempot_diagram.png")
print("  Ef_all_charge_S_poor.png")
print("  Ef_all_charge_S_rich.png")
