import sys
from power import power
reps = int(sys.argv[1])
for n in (250, 300, 350):
    for rho in (0.0, 0.36, 0.6):
        out = []
        for delta in (0.0, 0.05, 0.06):
            a, b = power(n, delta, rho, reps, 7 + n + int(rho * 100) + int(delta * 1000))
            out.append(f"d{delta}: sign {a:.3f} flip {b:.3f}")
        print(f"n {n} rho {rho}: " + " | ".join(out), flush=True)
