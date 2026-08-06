import re
# G row 215
for l in open("rust_G.txt"):
    if l.startswith("[row 215]"):
        print("G row 215:", l.strip())
        break
# dof 215 pos
lines = [l.split() for l in open("rust_dofpos.txt") if l.strip()]
print("dof 215 pos:", lines[215][:3])
# A row 215 (elim) — show its nonzeros
for l in open("rust_A_elim.txt"):
    if l.startswith("[row 215]"):
        print("A row 215:", l.strip()[:200])
        break
