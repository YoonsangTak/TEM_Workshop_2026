#!/usr/bin/env python3
"""
Defect formation energy analysis for Si:S_Si and Si:V_Si.

Produces 3 PNG images:
  1) chempot_diagram.png   - Delta mu_S allowed range (S-poor <-> S-rich)
  2) Ef_S_poor.png         - formation energy vs E_F at the S-poor limit
  3) Ef_S_rich.png         - formation energy vs E_F at the S-rich limit

For each defect the LOWER ENVELOPE over charge states is drawn (the physically
stable charge state at each E_F). Where two charge-state lines cross is the
charge transition level epsilon(q/q'), marked on the plot.

E_f(q, E_F) = E_defect(q) - E_perfect + chem_term + q*(E_VBM + E_F)
  chem_term(S_Si) = mu_Si - mu_S    ;   chem_term(V_Si) = mu_Si
assume_isolated='mp' (Makov-Payne) assumed already in E_defect.

Headless. Run from inside 2_defect_formation_energy/:
    python3 plot_formation_energy.py
"""

import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RY2EV = 13.605693122994

# ---------------------------------------------------------------- paths
BASE = "."
PATHS = {
    "bulk_Si":  f"{BASE}/1_Si/si.vc-relax.out",
    "SiS2":     f"{BASE}/2_SiS2/sis2.vc-relax.out",
    "S2":       f"{BASE}/3_S2/s2.relax.out",
    "perfect":  f"{BASE}/4_Si_bulk_supercell/si_bulk.scf.out",
    "S_Si_q0":  f"{BASE}/5_S_Si_substitution/charge_0/s_si.relax.out",
    "S_Si_q2":  f"{BASE}/5_S_Si_substitution/charge_2/s_si.relax.out",
    "V_Si_q0":  f"{BASE}/6_V_Si_vacancy/charge_0/v_si.relax.out",
    "V_Si_qm2": f"{BASE}/6_V_Si_vacancy/charge_-2/v_si.relax.out",
}

N_SI_BULK_ATOMS = 8
N_SIS2_FU       = 4
N_S2_ATOMS      = 2
BAND_GAP        = 0.6   # eV (edit to your value)

# ---------------------------------------------------------------- parsers
def read_total_energy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing output: {path}")
    e = None
    with open(path) as f:
        for line in f:
            if line.strip().startswith("!"):
                m = re.search(r"=\s*(-?\d+\.\d+)", line)
                if m:
                    e = float(m.group(1))
    if e is None:
        raise ValueError(f"No '!' total energy in {path}")
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

# ---------------------------------------------------------------- references
E_bulk_Si = read_total_energy(PATHS["bulk_Si"]) * RY2EV
mu_Si     = E_bulk_Si / N_SI_BULK_ATOMS
E_SiS2_fu = read_total_energy(PATHS["SiS2"]) * RY2EV / N_SIS2_FU
mu_S_ref  = read_total_energy(PATHS["S2"]) * RY2EV / N_S2_ATOMS

dmu_S_rich = 0.0
dmu_S_poor = (E_SiS2_fu - mu_Si) / 2.0 - mu_S_ref
mu_S_at = {"S-rich": mu_S_ref + dmu_S_rich, "S-poor": mu_S_ref + dmu_S_poor}

print(f"mu_Si           = {mu_Si:.4f} eV/atom")
print(f"E_SiS2_fu       = {E_SiS2_fu:.4f} eV/f.u.")
print(f"mu_S_ref (S2/2) = {mu_S_ref:.4f} eV")
print(f"Delta mu_S range: [{dmu_S_poor:.4f}, {dmu_S_rich:.4f}] eV")

E_perfect = read_total_energy(PATHS["perfect"]) * RY2EV
E_VBM     = read_vbm(PATHS["perfect"])
print(f"E_perfect       = {E_perfect:.4f} eV")
print(f"E_VBM           = {E_VBM:.4f} eV")

E = {k: read_total_energy(PATHS[k]) * RY2EV
     for k in ("S_Si_q0", "S_Si_q2", "V_Si_q0", "V_Si_qm2")}

# ---------------------------------------------------------------- model
# defect -> list of (q, E_defect, color), plus chem term builder
DEFECTS = {
    "S_Si": {
        "states": [(0, E["S_Si_q0"]), (2, E["S_Si_q2"])],
        "color": "tab:red",
        "label": "S$_{Si}$",
        "chem": lambda mu_S: mu_Si - mu_S,
    },
    "V_Si": {
        "states": [(0, E["V_Si_q0"]), (-2, E["V_Si_qm2"])],
        "color": "tab:blue",
        "label": "V$_{Si}$",
        "chem": lambda mu_S: mu_Si,
    },
}

def Ef(E_def, chem, q, EF):
    return E_def - E_perfect + chem + q * (E_VBM + EF)

def transition_levels(states, chem):
    """Return list of (EF_transition, q_from, q_to) within [0, gap]."""
    # sort by charge descending (more positive = more stable at low EF)
    s = sorted(states, key=lambda x: -x[0])
    levels = []
    for i in range(len(s) - 1):
        q1, e1 = s[i]
        q2, e2 = s[i + 1]
        # Ef equal: e1 - Eperf + chem + q1(Evbm+EF) = e2 - Eperf + chem + q2(Evbm+EF)
        # => (q1-q2)*EF = (e2 - e1) + (q2-q1)*Evbm
        if q1 == q2:
            continue
        EF_t = ((e2 - e1) + (q2 - q1) * E_VBM) / (q1 - q2)
        levels.append((EF_t, q1, q2))
    return levels

# ================================================================ IMAGE 1
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hlines(0, dmu_S_poor, dmu_S_rich, color="tab:green", lw=4)
ax.plot(dmu_S_rich, 0, "o", color="tab:green", ms=11)
ax.plot(dmu_S_poor, 0, "o", color="tab:orange", ms=11)
ax.annotate("S-rich\n($\\Delta\\mu_S=0$)", (dmu_S_rich, 0),
            textcoords="offset points", xytext=(-10, 18), ha="right", color="tab:green")
ax.annotate(f"S-poor\n($\\Delta\\mu_S={dmu_S_poor:.2f}$)", (dmu_S_poor, 0),
            textcoords="offset points", xytext=(10, -34), ha="left", color="tab:orange")
pad = 0.1 * abs(dmu_S_poor) if dmu_S_poor != 0 else 0.2
ax.set_xlim(dmu_S_poor - pad, dmu_S_rich + pad)
ax.set_ylim(-1, 1); ax.set_yticks([])
ax.set_xlabel("$\\Delta\\mu_S$ (eV)")
ax.set_title("Allowed S chemical potential range")
for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("chempot_diagram.png", dpi=150); plt.close()
print("Saved: chempot_diagram.png")

# ================================================================ IMAGE 2 & 3
EF = np.linspace(0.0, BAND_GAP, 400)
for limit in ("S-poor", "S-rich"):
    mu_S = mu_S_at[limit]
    plt.figure(figsize=(7, 5.5))
    for name, d in DEFECTS.items():
        chem = d["chem"](mu_S)
        # lower envelope over charge states
        all_lines = np.array([[Ef(e_def, chem, q, ef) for ef in EF]
                              for (q, e_def) in d["states"]])
        envelope = all_lines.min(axis=0)
        plt.plot(EF, envelope, color=d["color"], lw=2.2, label=d["label"])
        # faint individual charge lines for reference
        for (q, e_def), ln in zip(d["states"], all_lines):
            plt.plot(EF, ln, color=d["color"], lw=0.7, ls="--", alpha=0.4)
        # mark charge transition levels inside the gap
        for EF_t, q1, q2 in transition_levels(d["states"], chem):
            if 0 <= EF_t <= BAND_GAP:
                y = Ef(d["states"][0][1], chem, q1, EF_t) if False else \
                    Ef([e for (qq, e) in d["states"] if qq == q1][0], chem, q1, EF_t)
                plt.plot(EF_t, y, "k.", ms=9, zorder=5)
                plt.annotate(f"$\\varepsilon$({q1:+d}/{q2:+d})", (EF_t, y),
                             textcoords="offset points", xytext=(4, 6), fontsize=9)
    plt.axvline(0.0, color="gray", ls=":", lw=0.8)
    plt.axvline(BAND_GAP, color="gray", ls=":", lw=0.8)
    plt.xlabel("Fermi level $E_F$ (eV)   [0 = VBM]")
    plt.ylabel("Formation energy $E_f$ (eV)")
    plt.title(f"Defect formation energy in Si ({limit} limit)")
    plt.legend(); plt.grid(alpha=0.3); plt.xlim(0, BAND_GAP)
    plt.tight_layout()
    fname = f"Ef_{limit.replace('-', '_')}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved: {fname}")

# print transition levels
print("\nCharge transition levels (eV above VBM):")
for name, d in DEFECTS.items():
    for EF_t, q1, q2 in transition_levels(d["states"], 0.0):
        inside = "(in gap)" if 0 <= EF_t <= BAND_GAP else "(outside gap)"
        print(f"  {name}  eps({q1:+d}/{q2:+d}) = {EF_t:.3f} eV  {inside}")

print("\nDone. 3 images created.")
