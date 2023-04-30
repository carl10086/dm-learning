import zlib


def crc32(filename, chunksize=65536):
    """Compute the CRC-32 checksum of the contents of the given filename"""
    with open(filename, "rb") as f:
        checksum = 0
        while chunk := f.read(chunksize):
            checksum = zlib.crc32(chunk, checksum)
        return checksum


if __name__ == '__main__':
    filename = '/root/autodl-tmp/models/ui_models_1/rev_animated/Rev_Animated_v1.2.2_Pruned.safetensors'
    crc32 = crc32(filename)
    # 15467F52
    print(f"crc32, {hex(crc32)}")
