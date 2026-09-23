def sets(p):
    out=[]
    for ln in open(p):
        t=ln.split()
        if len(t)==10 and t[1]=='5':
            out.append((t[0], tuple(sorted(map(int,t[2:])))))
    return sorted(out)
a=sets('reflected_topo.txt'); b=sets('cpp_reflected_topo.txt')
print('rust elems:',len(a),'cpp elems:',len(b))
print('IDENTICAL' if a==b else 'DIFFER')
if a!=b:
    sa=set(a); sb=set(b)
    print('only-rust:',sorted(sa-sb)); print('only-cpp:',sorted(sb-sa))
