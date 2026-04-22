import hashlib

PRESET_COLORS = {
    "physical_damage":    (0, 0, 255),      # Kırmızı
    "bird_drop":          (0, 140, 255),    # Turuncu
    "dust_particle":      (0, 255, 255),    # Sarı
    "leaf":               (0, 200, 0),      # Yeşil
    "snow":               (255, 255, 255),  # Beyaz
    "healthy":            (0, 255, 0),      # Parlak yeşil
    "bird_feather":       (255, 100, 0),    # Mavi
}

def generate_class_colors(class_names: dict[int, str]) -> dict[int, tuple[int, int, int]]:
    """Get color for each class based on dictionary."""
    colors = {}
    for cls_id, cls_name in class_names.items():
        if cls_name in PRESET_COLORS:
            colors[cls_id] = PRESET_COLORS[cls_name]
        else:
            # Deterministic pseudo-random color based on hash of class name
            hash_code = hashlib.md5(cls_name.encode('utf-8')).hexdigest()
            r = int(hash_code[:2], 16)
            g = int(hash_code[2:4], 16)
            b = int(hash_code[4:6], 16)
            # Make sure it's somewhat bright but unpredictable
            colors[cls_id] = (b, g, max(r, 100))
    return colors
