with open('crates/assembly/src/discrete_op.rs', 'r', encoding='utf-8') as f:
    lines = f.readlines()
out = []
for line in lines:
    if '3D discrete operator needs debugging' in line:
        continue
    out.append(line)
with open('crates/assembly/src/discrete_op.rs', 'w', encoding='utf-8') as f:
    f.writelines(out)
print('stripped')
