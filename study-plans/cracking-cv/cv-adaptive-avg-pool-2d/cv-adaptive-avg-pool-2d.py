import math

def adaptive_avg_pool_2d(image, output_h, output_w):
    """
    Returns: 2D Python list of shape (output_h, output_w), values rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    result = []
    for i in range(output_h):
        start_h = (i * H) // output_h
        end_h = math.ceil((i + 1) * H / output_h)

        row = []
        for j in range(output_w):
            start_w = (j * W) // output_w
            end_w = math.ceil((j + 1) * W / output_w)

            total = 0.0
            count = 0
            for r in range(start_h, end_h):
                for c in range(start_w, end_w):
                    total += image[r][c]
                    count += 1

            avg = total / count
            row.append(round(float(avg), 4))
        result.append(row)

    return result