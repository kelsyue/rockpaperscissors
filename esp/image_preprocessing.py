"""
image_preprocessing.py - BMP Image Preprocessing Utilities (ESP32-S3)
======================================================================
Rubric:
    - "CNN running on the ESP" (20 pts)
    - "CNN running on the ESP able to classify video images in real(ish)
       time" (40 pts)

This module provides image preprocessing functions that run directly on
the ESP32-S3 microcontroller. It resizes and thresholds raw 96x96 BMP
images from the camera into 32x32 binary images suitable for the CNN,
and strips the BMP header to extract raw pixel bytes for model inference.

These functions must produce output identical to the preprocessing done
on the laptop during training - any mismatch will degrade model accuracy.

Source: Provided by course instructor 
        Minor formatting and comments added by student.
        Original BMP manipulation logic written by course staff.
"""


def resize_96x96_to_32x32_and_threshold(bmp_data_mv, threshold, inversion=False):
    """
    Resizes a 96x96 grayscale BMP to 32x32 and applies a binary threshold.

    This is the primary preprocessing function used before CNN inference.
    It maps each output pixel to the nearest input pixel (nearest-neighbor
    downsampling), then converts the grayscale value to pure black or white
    based on the threshold.

    This function is called on every frame in classify.py before inference,
    and its output must match the preprocessing done during laptop training.

    Args:
        bmp_data_mv: raw BMP bytearray including headers (96x96 grayscale)
        threshold (int): pixel cutoff value (0-255). pixels >= threshold
                         become white (255), pixels below become black (0).
                         pass -1 to skip thresholding and keep grayscale.
        inversion (bool): if True, reverses black/white output. default False.

    Returns:
        bytearray: new 32x32 BMP bytearray including headers and palette
    """
    bmp_data = bytearray(bmp_data_mv)

    OLD_WIDTH = 96
    OLD_HEIGHT = 96

    NEW_WIDTH = 32
    NEW_HEIGHT = 32

    OLD_BMP_HEADER_SIZE = 14 
    OLD_DIB_HEADER_SIZE = 40  
    OLD_PALETTE_SIZE = 256 * 4  

    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4

    NEW_ROW_PADDING = (NEW_WIDTH % 4)

    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )

    new_bmp_data = bytearray(new_file_size)

    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    new_bmp_data[0:2] = b'BM'  
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')
    new_bmp_data[26:28] = b'\x01\x00' 
    new_bmp_data[28:30] = b'\x08\x00'  
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            old_x = (new_x * OLD_WIDTH) // NEW_WIDTH
            old_y = (new_y * OLD_HEIGHT) // NEW_HEIGHT

            old_y = OLD_HEIGHT - 1 - old_y

            old_pixel_offset = old_pixel_data_offset + old_y * (OLD_WIDTH + (OLD_WIDTH % 4)) + old_x

            pixel_value = bmp_data[old_pixel_offset] & 0xFF

            if threshold >= 0:
                if inversion:
                    pixel_value = 0 if pixel_value >= threshold else 255
                else:
                    pixel_value = 255 if pixel_value >= threshold else 0

            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1

        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data


def resize_96x96_to_32x32(bmp_data_mv):
    """
    Resizes a 96x96 grayscale BMP to 32x32 without thresholding.

    Preserves all grayscale values (0-255) instead of converting to
    black/white. Useful for visualization or when full grayscale is needed.

    Args:
        bmp_data_mv: raw BMP bytearray including headers (96x96 grayscale)

    Returns:
        bytearray: new 32x32 BMP bytearray with full grayscale values
    """
    bmp_data = bytearray(bmp_data_mv)
    OLD_WIDTH = 96
    OLD_HEIGHT = 96
    NEW_WIDTH = 32
    NEW_HEIGHT = 32
    OLD_BMP_HEADER_SIZE = 14
    OLD_DIB_HEADER_SIZE = 40
    OLD_PALETTE_SIZE = 256 * 4
    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4
    NEW_ROW_PADDING = (NEW_WIDTH % 4)
    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )
    new_bmp_data = bytearray(new_file_size)

    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    new_bmp_data[0:2] = b'BM'
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')
    new_bmp_data[26:28] = b'\x01\x00'
    new_bmp_data[28:30] = b'\x08\x00'
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            old_x = (new_x * OLD_WIDTH) // NEW_WIDTH
            old_y = (new_y * OLD_HEIGHT) // NEW_HEIGHT
            old_y = OLD_HEIGHT - 1 - old_y 
            old_pixel_offset = old_pixel_data_offset + old_y * (OLD_WIDTH + (OLD_WIDTH % 4)) + old_x
            pixel_value = bmp_data[old_pixel_offset] & 0xFF
            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1
        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data


def strip_bmp_header(bmp_byte_array):
    """
    Strips the BMP file header and color palette, returning only raw pixel bytes.

    The emlearn CNN model expects a flat array of 1024 raw pixel values
    (32x32 = 1024), not a full BMP file. This function removes the header
    and palette so only the pixel data remains.

    BMP structure for 32x32 8-bit grayscale:
        14 bytes  - BMP file header
        40 bytes  - DIB info header
        1024 bytes - 256-color palette (256 x 4 bytes each)
        1024 bytes - pixel data (32 x 32 pixels)

    Args:
        bmp_byte_array: bytearray of a 32x32 8-bit grayscale BMP file

    Returns:
        bytes: 1024 raw pixel bytes (32x32), ready for model inference

    Raises:
        ValueError: if the input is too small or pixel data is not 32x32
    """
    BMP_HEADER_SIZE = 54     
    PALETTE_SIZE = 256 * 4  

    if len(bmp_byte_array) <= BMP_HEADER_SIZE + PALETTE_SIZE:
        raise ValueError("Invalid BMP file: too small.")

    pixel_data_start = BMP_HEADER_SIZE + PALETTE_SIZE
    pixel_data = bmp_byte_array[pixel_data_start:]

    if len(pixel_data) != 32 * 32:
        raise ValueError("Invalid BMP file: pixel data is not 32x32.")

    return pixel_data