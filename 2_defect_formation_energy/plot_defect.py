#!/usr/bin/env python3
"""
========================================================================
 Defect formation energy diagram for Si:S_Si and Si:V_Si  (QUALITATIVE)
========================================================================

Reads Quantum ESPRESSO output files, computes defect formation energies
as a function of the Fermi level, and saves three figures:

  1) chempot_diagram.png   Delta mu_S allowed range (S-poor <-> S-rich)
  2) Ef_S_poor.png         formation energy vs E_F at the S-poor limit
  3) Ef_S_rich.png         formation energy vs E_F at the S-rich limit

------------------------------------------------------------------------
 FORMATION ENERGY
------------------------------------------------------------------------
   E_f(q, E_F) = E_def(q) - E_perfect + chem + q*(E_VBM + E_F)

   chem(S_Si) = mu_Si - mu_S      (remove one Si, add one S)
   chem(V_Si) = mu_Si             (remove one Si)

   mu_Si  : from bulk Si              (E / N_atoms)
   mu_S   : S-poor  -> bulk-Si limit (set by SiS2 equilibrium)
            S-rich  -> S2 reference   (Delta mu_S = 0)
   E_VBM  : highest occupied level of the perfect supercell
   q      : charge state; the line slope equals q

For each defect the LOWER ENVELOPE over its charge states is drawn.
For S_Si, charge states q = 0, +1, +2 are included.
For V_Si, charge states q = 0, -2 are included.

------------------------------------------------------------------------
 IMPORTANT: this is a QUALITATIVE diagram
------------------------------------------------------------------------
 * All energies are RAW total energies at the SAME ecut.
 * No finite-size charged-defect correction is applied.
 * QE's "Total+Makov-Payne energy" lines are deliberately ignored.
 * Absolute formation energies and charge transition levels are not
   quantitative without proper charged-defect correction, potential
   alignment, and band-gap correction.
========================================================================
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RY2EV = 13.605693122994

# ----------------------------------------------------------------- paths
BASE = "."
PATHS = {
    "bulk_Si":  f"{BASE}/1_Si/si.vc-relax.out",
    "SiS2":     f"{BASE}/2_SiS2/sis2.vc-relax.out",
    "S2":       f"{BASE}/3_S2/s2.relax.out",
    "perfect":  f"{BASE}/4_Si_bulk_supercell/si_bulk.scf.out",

    # S substitutional defect
    "S_Si_q0":  f"{BASE}/5_S_Si_substitution/charge_0/s_si.relax.out",
    "S_Si_q1":  f"{BASE}/5_S_Si_substitution/charge_1/s_si.relax.out",
    "S_Si_q2":  f"{BASE}/5_S_Si_substitution/charge_2/s_si.relax.out",

    # Si vacancy
    "V_Si_q0":  f"{BASE}/6_V_Si_vacancy/charge_0/v_si.relax.out",
    "V_Si_qm2": f"{BASE}/6_V_Si_vacancy/charge_-2/v_si.relax.out",
}

N_SI_BULK_ATOMS = 8     # 1_Si conventional cell = 8; use 2 if primitive
N_SIS2_FU       = 4     # 12-atom SiS2 = 4 formula units
N_S2_ATOMS      = 2
BAND_GAP        = 0.6   # eV, PBE Si gap; edit to your computed value

# ----------------------------------------------------------------- parser
def read_total_energy(path):
    """
    Read the last raw '! total energy' line in Ry.

    Important:
    QE may print both:
      ! total energy
      ! Total+Makov-Payne energy

    Here we intentionally ignore Makov-Payne lines.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing output: {path}")

    e = None
    with open(path) as f:
        for line in f:
            t = line.strip()
            if t.startswith("!") and "Makov" not in t and "total energy" in t:
                m = re.search(r"=\s*(-?\d+\.\d+)", t)
                if m:
                    e = float(m.group(1))

    if e is None:
        raise ValueError(f"No raw '! total energy' in {path}")

    return e


def read_vbm(path):
    """
    Read highest occupied level from perfect supercell output.
    If not found, fall back to Fermi energy.
    """
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
    """Pretty charge label for plot annotations."""
    if q > 0:
        return f"+{q}"
    if q < 0:
        return f"{q}"
    return "0"


# ----------------------------------------------------------------- references
E_bulk_Si = read_total_energy(PATHS["bulk_Si"]) * RY2EV
mu_Si     = E_bulk_Si / N_SI_BULK_ATOMS

E_SiS2_fu = read_total_energy(PATHS["SiS2"]) * RY2EV / N_SIS2_FU
mu_S_ref  = read_total_energy(PATHS["S2"]) * RY2EV / N_S2_ATOMS

# Delta mu_S = mu_S - mu_S_ref  <= 0
# S-rich: Delta mu_S = 0
# S-poor: mu_Si fixed to bulk Si, SiS2 equilibrium
dmu_S_rich = 0.0
dmu_S_poor = (E_SiS2_fu - mu_Si) / 2.0 - mu_S_ref

mu_S_at = {
    "S-rich": mu_S_ref + dmu_S_rich,
    "S-poor": mu_S_ref + dmu_S_poor,
}

print(f"mu_Si           = {mu_Si:.4f} eV/atom")
print(f"E_SiS2_fu       = {E_SiS2_fu:.4f} eV/f.u.")
print(f"mu_S_ref (S2/2) = {mu_S_ref:.4f} eV")
print(f"Delta mu_S range: [{dmu_S_poor:.4f}, {dmu_S_rich:.4f}] eV")

E_perfect = read_total_energy(PATHS["perfect"]) * RY2EV
E_VBM     = read_vbm(PATHS["perfect"])

print(f"E_perfect       = {E_perfect:.4f} eV")
print(f"E_VBM           = {E_VBM:.4f} eV")

# Defect total energies
E = {
    k: read_total_energy(PATHS[k]) * RY2EV
    for k in (
        "S_Si_q0",
        "S_Si_q1",
        "S_Si_q2",
        "V_Si_q0",
        "V_Si_qm2",
    )
}

print("\nDefect total energies:")
for k, v in E.items():
    print(f"  {k:10s} = {v:.6f} eV")

# ----------------------------------------------------------------- model
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
    Defect formation energy at Fermi level EF.
    EF is measured from VBM.
    """
    return E[key] - E_perfect + chem + q * (E_VBM + EF)


def transition_levels(states):
    """
    Return charge transition levels between adjacent charge states.

    states: list of (q, energy_key)
    output: list of (EF_transition, q1, q2, key1)

    The chemical potential term cancels for charge transition levels
    of the same defect.
    """
    s = sorted(states, key=lambda x: -x[0])  # high charge -> low charge
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


# ================================================================ IMAGE 1
# Chemical potential diagram
fig, ax = plt.subplots(figsize=(7, 3.2))

ax.hlines(0, dmu_S_poor, dmu_S_rich, color="tab:green", lw=4)
ax.plot(dmu_S_rich, 0, "o", color="tab:green", ms=11)
ax.plot(dmu_S_poor, 0, "o", color="tab:orange", ms=11)

ax.annotate(
    "S-rich\n($\\Delta\\mu_S=0$)",
    (dmu_S_rich, 0),
    textcoords="offset points",
    xytext=(-10, 18),
    ha="right",
    color="tab:green",
)

ax.annotate(
    f"S-poor\n($\\Delta\\mu_S={dmu_S_poor:.2f}$)",
    (dmu_S_poor, 0),
    textcoords="offset points",
    xytext=(10, -34),
    ha="left",
    color="tab:orange",
)

pad = 0.1 * abs(dmu_S_poor) if dmu_S_poor != 0 else 0.2
ax.set_xlim(dmu_S_poor - pad, dmu_S_rich + pad)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xlabel("$\\Delta\\mu_S$ (eV)")
ax.set_title("Allowed S chemical potential range")

for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("chempot_diagram.png", dpi=150)
plt.close()

print("Saved: chempot_diagram.png")


# ================================================================ IMAGE 2 & 3
# Formation energy diagrams
EF = np.linspace(0.0, BAND_GAP, 400)

for limit in ("S-poor", "S-rich"):
    mu_S = mu_S_at[limit]

    plt.figure(figsize=(7, 5.5))

    for name, d in DEFECTS.items():
        chem = d["chem"](mu_S)

        # all charge-state lines
        lines = np.array([
            [Ef(key, chem, q, ef) for ef in EF]
            for (q, key) in d["states"]
        ])

        # lower envelope
        envelope = lines.min(axis=0)

        plt.plot(
            EF,
            envelope,
            color=d["color"],
            lw=2.4,
            label=d["label"],
        )

        # individual charge-state lines
        for (q, key), ln in zip(d["states"], lines):
            plt.plot(
                EF,
                ln,
                color=d["color"],
                lw=0.8,
                ls="--",
                alpha=0.45,
            )

            # optional charge label near right side
            x_text = BAND_GAP * 0.92
            y_text = Ef(key, chem, q, x_text)
            plt.text(
                x_text,
                y_text,
                f"q={charge_label(q)}",
                color=d["color"],
                fontsize=8,
                alpha=0.7,
                va="center",
            )

        # charge transition levels
        for EF_t, q1, q2, k1 in transition_levels(d["states"]):
            if 0 <= EF_t <= BAND_GAP:
                y = Ef(k1, chem, q1, EF_t)

                plt.plot(EF_t, y, "k.", ms=9, zorder=5)

                plt.annotate(
                    f"$\\varepsilon$({charge_label(q1)}/{charge_label(q2)})={EF_t:.2f}",
                    (EF_t, y),
                    textcoords="offset points",
                    xytext=(4, 6),
                    fontsize=9,
                )

    plt.axvline(0.0, color="gray", ls=":", lw=0.8)
    plt.axvline(BAND_GAP, color="gray", ls=":", lw=0.8)

    plt.xlabel("Fermi level $E_F$ (eV)   [0 = VBM]")
    plt.ylabel("Formation energy $E_f$ (eV)")
    plt.title(f"Defect formation energy in Si ({limit} limit)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, BAND_GAP)

    plt.tight_layout()

    fname = f"Ef_{limit.replace('-', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"Saved: {fname}")


# ----------------------------------------------------------------- transition levels
print("\nCharge transition levels (eV above VBM):")

for name, d in DEFECTS.items():
    for EF_t, q1, q2, k1 in transition_levels(d["states"]):
        tag = "(in gap)" if 0 <= EF_t <= BAND_GAP else "(OUTSIDE gap)"
        print(
            f"  {name:5s}  eps({charge_label(q1)}/{charge_label(q2)}) "
            f"= {EF_t:.3f} eV  {tag}"
        )

print("\nDone. 3 images created.")

print("Note: qualitative raw PBE diagram; no finite-size charge correction.")
