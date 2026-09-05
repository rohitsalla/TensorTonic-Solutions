import torch

def multimodal_rope_coordinates(segments):
    """
    Returns: dictionary containing CPU int64 RoPE coordinates
    """
    time_coords   = []
    height_coords = []
    width_coords  = []
    offset = 0

    for seg in segments:
        stype = seg["type"]

        if stype == "text":
            L = seg["length"]
            for j in range(L):
                p = offset + j
                time_coords.append(p)
                height_coords.append(p)
                width_coords.append(p)
            offset += L

        elif stype == "separator":
            time_coords.append(offset)
            height_coords.append(offset)
            width_coords.append(offset)
            offset += 1

        elif stype == "image":
            H, W = seg["height"], seg["width"]
            for r in range(H):
                for c in range(W):
                    time_coords.append(offset)
                    height_coords.append(offset + r)
                    width_coords.append(offset + c)
            offset += max(H, W)

        elif stype == "video":
            F  = seg["frames"]
            H  = seg["height"]
            W  = seg["width"]
            s  = seg["frame_stride"]
            for f in range(F):
                for r in range(H):
                    for c in range(W):
                        time_coords.append(offset + f * s)
                        height_coords.append(offset + r)
                        width_coords.append(offset + c)
            advance = max((F - 1) * s + 1, H, W) if F > 0 else max(H, W)
            offset += advance

    coords = torch.tensor(
        [time_coords, height_coords, width_coords],
        dtype=torch.int64
    )

    return {"coordinates": coords}