"""Scan identity groups used to prevent near-duplicate partition leakage."""


def leakage_group_ids(rows):
    if 'leakage_group_id' not in rows:
        return rows.patient_id.astype(str)
    groups = rows.leakage_group_id.astype(str).str.strip()
    return groups.where(groups.ne(''), rows.patient_id.astype(str))


def check_identity_partitions(partitions, declared_groups=None):
    owners = {}
    declared = {}
    for group, members in (declared_groups or {}).items():
        for pid in members:
            pid = str(pid)
            if pid in declared and declared[pid] != str(group):
                raise ValueError(f'Conflicting saved identity groups for {pid}')
            declared[pid] = str(group)
    for name, rows in partitions.items():
        for pid, group in zip(rows.patient_id.astype(str), leakage_group_ids(rows)):
            keys = [('manifest', group)]
            if pid in declared:
                keys.append(('saved', declared[pid]))
            for other in keys:
                owner = owners.setdefault(other, name)
                if owner != name:
                    raise ValueError(f'Near-duplicate identity group leakage: {other} in {owner} and {name}')
