#!/usr/bin/env python3
"""
Plot charge transition levels of substitutional sulfur in Si.

Only S_Si charge states are shown:
    S_Si^0, S_Si^+1, S_Si^+2

No V_Si comparison in this script.
No charged-defect correction is applied.
QE "Total+Makov-Payne energy" lines are ignored.
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
    "bulk_Si": f"{BASE}/1_Si/si.vc-relax.out",
    "SiS2":    f"{BASE}/2_SiS2/sis2.vc-relax.out",
    "S2":      f"{BASE}/3_S2/s2.relax.out",
    "perfect": f"{BASE}/4_Si_bulk_supercell/si_bulk.scf.out",

    "S_Si_q0": f"{BASE}/5_S_Si_substitution/charge_0/s_si.relax.out",
    "S_Si_q1": f"{BASE}/5_S_Si_substitution/charge_1/s_si.relax.out",
    "S_Si_q2": f"{BASE}/5_S_Si_substitution/charge_2/s_si.relax.out",
}

N_SI_BULK_ATOMS = 8
N_SIS2_FU = 4
N_S2_ATOMS = 2

BAND_GAP = 0.60   # PBE Si gap in eV


# ============================================================
# Parsers
# ============================================================
def read_total_energy(path):
    """
    Read last raw QE total energy in Ry.
    Ignore 'Total+Makov-Payne energy'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    e = None
    with open(path) as f:
        for line in f:
            t = line.strip()
            if t.startswith("!") and "Makov" not in t and "total energy" in t:
                m = re.search(r"=\s*(-?\d+\.\d+)", t)
                if m:
                    e = float(m.group(1))

    if e is None:
        raise ValueError(f"No raw total energy found in {path}")

    return e


def read_vbm(path):
    with open(path) as f:
        txt = f.read()

    m = re.search(r"highest occupied.*?:\s*(-?\d+\.\d+)", txt)
    if m:
        return float(m.group(1))

    m = re.search(r"Fermi energy is\s*(-?\d+\.\d+)", txt)
    if m:
        return float(m.group(1))

    raise ValueError(f"Could not find VBM/Fermi in {path}")


def charge_label(q):
    if q > 0:
        return f"+{q}"
    if q < 0:
        return f"{q}"
    return "0"


# ============================================================
# Reference energies
# ============================================================
E_bulk_Si = read_total_energy(PATHS["bulk_Si"]) * RY2EV
mu_Si = E_bulk_Si / N_SI_BULK_ATOMS

E_SiS2_fu = read_total_energy(PATHS["SiS2"]) * RY2EV / N_SIS2_FU
mu_S_ref = read_total_energy(PATHS["S2"]) * RY2EV / N_S2_ATOMS

dmu_S_rich = 0.0
dmu_S_poor = (E_SiS2_fu - mu_Si) / 2.0 - mu_S_ref

mu_S_at = {
    "S-poor": mu_S_ref + dmu_S_poor,
    "S-rich": mu_S_ref + dmu_S_rich,
}

E_perfect = read_total_energy(PATHS["perfect"]) * RY2EV
E_VBM = read_vbm(PATHS["perfect"])

E = {
    "S_Si_q0": read_total_energy(PATHS["S_Si_q0"]) * RY2EV,
    "S_Si_q1": read_total_energy(PATHS["S_Si_q1"]) * RY2EV,
    "S_Si_q2": read_total_energy(PATHS["S_Si_q2"]) * RY2EV,
}

print("Reference energies")
print(f"  mu_Si           = {mu_Si:.6f} eV")
print(f"  mu_S_ref        = {mu_S_ref:.6f} eV")
print(f"  Delta mu_S poor = {dmu_S_poor:.6f} eV")
print(f"  E_perfect       = {E_perfect:.6f} eV")
print(f"  E_VBM           = {E_VBM:.6f} eV")
print()


# ============================================================
# Formation energy model
# ============================================================
STATES = [
    (0, "S_Si_q0"),
    (1, "S_Si_q1"),
    (2, "S_Si_q2"),
]

def Ef(key, chem, q, EF):
    return E[key] - E_perfect + chem + q * (E_VBM + EF)


def transition_levels(states):
    """
    Adjacent charge transition levels:
        +2/+1 and +1/0
    """
    s = sorted(states, key=lambda x: -x[0])
    out = []

    for i in range(len(s) - 1):
        q1, k1 = s[i]
        q2, k2 = s[i + 1]

        a1 = E[k1] + q1 * E_VBM
        a2 = E[k2] + q2 * E_VBM

        EF_t = (a2 - a1) / (q1 - q2)
        out.append((EF_t, q1, q2, k1))

    return out


# ============================================================
# Plot
# ============================================================
EF = np.linspace(0.0, BAND_GAP, 400)

for limit in ("S-poor", "S-rich"):
    mu_S = mu_S_at[limit]
    chem = mu_Si - mu_S

    plt.figure(figsize=(7, 5.5))

    lines = []
    for q, key in STATES:
        y = np.array([Ef(key, chem, q, ef) for ef in EF])
        lines.append(y)

        plt.plot(
            EF,
            y,
            lw=1.4,
            ls="--",
            alpha=0.75,
            label=f"S$_{{Si}}^{{{charge_label(q)}}}$"
        )

        x_text = BAND_GAP * 0.92
        y_text = Ef(key, chem, q, x_text)
        plt.text(
            x_text,
            y_text,
            f"q={charge_label(q)}",
            fontsize=9,
            va="center"
        )

    lines = np.array(lines)
    envelope = lines.min(axis=0)

    plt.plot(
        EF,
        envelope,
        color="black",
        lw=2.5,
        label="lower envelope"
    )

    for EF_t, q1, q2, k1 in transition_levels(STATES):
        tag = "in gap" if 0 <= EF_t <= BAND_GAP else "outside gap"
        print(
            f"{limit}: S_Si eps({charge_label(q1)}/{charge_label(q2)}) "
            f"= {EF_t:.4f} eV  ({tag})"
        )

        if 0 <= EF_t <= BAND_GAP:
            y_t = Ef(k1, chem, q1, EF_t)
            plt.plot(EF_t, y_t, "ko", ms=5)
            plt.annotate(
                f"$\\varepsilon$({charge_label(q1)}/{charge_label(q2)})={EF_t:.2f} eV",
                (EF_t, y_t),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9
            )

    plt.axvline(0.0, color="gray", ls=":", lw=0.8)
    plt.axvline(BAND_GAP, color="gray", ls=":", lw=0.8)

    plt.xlabel("Fermi level $E_F$ (eV)   [0 = VBM]")
    plt.ylabel("Formation energy $E_f$ (eV)")
    plt.title(f"S$_{{Si}}$ charge transition levels in Si ({limit})")
    plt.xlim(0.0, BAND_GAP)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = f"S_Si_CTL_{limit.replace('-', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"Saved: {fname}")

print("\nDone.")
