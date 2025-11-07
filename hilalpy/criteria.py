def mabims(height, elongation):
    if height >= 3 and elongation >= 6.4:
        return "Hilal Mungkin Terlihat (MABIMS)"
    else:
        return "Hilal Tidak Terlihat (MABIMS)"

def yallop(q):
    if q is None:
        return "Error: nilai q tidak tersedia"
    if q >= 0.216:
        return "Hilal Terlihat (Yallop)"
    elif q >= -0.014:
        return "Hilal Mungkin Terlihat (Yallop)"
    elif q >= -0.160:
        return "Memerlukan alat optik (Yallop)"
    elif q >= -0.232:
        return "Hanya dengan teleskop (Yallop)"
    else:
        return "Hilal Tidak Terlihat (Yallop)"