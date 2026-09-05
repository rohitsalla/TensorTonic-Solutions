import torch

def vision_transformer_patchify(images, patch_height, patch_width):
    """
    Returns: dictionary containing patch tokens and grid coordinates
    """
    B, C, H, W = images.shape
    gh = H // patch_height   # grid rows
    gw = W // patch_width    # grid columns

    # Reshape: (B, C, gh, ph, gw, pw) then permute to (B, gh, gw, C, ph, pw)
    x = images.reshape(B, C, gh, patch_height, gw, patch_width)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # (B, gh, gw, C, ph, pw)

    # Flatten patch count and patch content dimensions
    tokens = x.reshape(B, gh * gw, C * patch_height * patch_width)

    # Row-major grid coordinates: (gh*gw, 2)
    rows = torch.arange(gh, device=images.device).repeat_interleave(gw)
    cols = torch.arange(gw, device=images.device).repeat(gh)
    coordinates = torch.stack([rows, cols], dim=1).to(torch.int64)

    return {"tokens": tokens, "coordinates": coordinates}