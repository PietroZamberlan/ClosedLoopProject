import numpy as np
import cairo

from pystim import utils


name = 'test'

default_configuration = {}


def generate(args):

    configuration = utils.handle_arguments_and_configurations(name, args)

    # dmd_configuration = ...

    width = 1920  # px  # fixed by the DMD
    height = 1080  # px  # fixed by the DMD
    scale = 2.0  # px / µm

    fod_width = float(width) / scale  # µm  # width of the field of display
    fod_height = float(height) / scale  # µm  # height of the field of display

    # Create surface.
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    context = cairo.Context(surface)
    # Transform space.
    tx, ty = float(width) / 2.0, float(height) / 2.0
    context.translate(tx, ty)
    sx, sy = +scale, scale
    context.scale(sx, sy)
    # Set background.
    x_min = - fod_width / 2.0
    x_max = + fod_width / 2.0
    y_min = + fod_height / 2.0
    y_max = - fod_height / 2.0
    context.set_source_rgb(1.0, 1.0, 1.0)  # i.e. white
    context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
    context.fill()

    nb_electrodes = 16
    inter_electrode_distance = 30.0  # µm

    # Draw MEA contour.
    mea_size = float(nb_electrodes) * inter_electrode_distance
    contour_width = inter_electrode_distance / 2.0
    x_min = - mea_size / 2.0 - contour_width
    x_max = + mea_size / 2.0 + contour_width
    y_min = + mea_size / 2.0 + contour_width
    y_max = - mea_size / 2.0 - contour_width
    context.set_source_rgb(0.0, 0.0, 0.0)  # i.e. black
    context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
    context.fill()
    x_min = - mea_size / 2.0
    x_max = + mea_size / 2.0
    y_min = + mea_size / 2.0
    y_max = - mea_size / 2.0
    context.set_source_rgb(1.0, 1.0, 1.0)  # i.e. white
    context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
    context.fill()

    # Draw horizontal line.
    line_width = inter_electrode_distance / 4.0
    x_min = - fod_width / 2.0
    x_max = + fod_width / 2.0
    y_min = + line_width / 2.0
    y_max = - line_width / 2.0
    context.set_source_rgb(0.0, 0.0, 0.0)  # i.e. black
    context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
    context.fill()

    # Draw vertical line.
    line_width = inter_electrode_distance / 4.0
    x_min = - line_width / 2.0
    x_max = + line_width / 2.0
    y_min = + fod_height / 2.0
    y_max = - fod_height / 2.0
    context.set_source_rgb(0.0, 0.0, 0.0)  # i.e. black
    context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
    context.fill()

    # Draw text.
    for i in range(0, 16):
        for j in range(0, 16):
            text_string = "X"
            context.set_font_size(inter_electrode_distance / 2.0)
            # print(context.font_extents())
            font_ascent, _, _, _, _ = context.font_extents()
            # print(context.text_extents(text_string))
            text_x_bearing, text_y_bearing, text_width, _, _, _ = context.text_extents(text_string)
            tx = text_x_bearing + text_width / 2.0
            ty = 0.3 * font_ascent
            # x = - inter_electrode_distance / 2.0 - tx
            # y = - inter_electrode_distance / 2.0 - ty
            x = - mea_size / 2.0 + float(i) * inter_electrode_distance + inter_electrode_distance / 2.0 - tx
            y = + mea_size / 2.0 - float(j) * inter_electrode_distance - inter_electrode_distance / 2.0 + ty
            context.move_to(x, y)
            context.set_source_rgb(0.0, 0.0, 0.0)  # i.e. black
            context.show_text(text_string)

            a = 1.0
            x_min = x - a + tx
            x_max = x + a + tx
            y_min = y + a - ty
            y_max = y - a - ty
            context.set_source_rgb(0.5, 0.5, 0.5)  # i.e. gray
            context.rectangle(x_min, y_min, x_max - x_min, y_max - y_min)
            context.fill()

    # Draw text.
    text_string = "I bet you no blind people will get this joke."
    context.set_font_size(inter_electrode_distance / 2.0)
    # print(context.font_extents())
    font_ascent, _, _, _, _ = context.font_extents()
    # print(context.text_extents(text_string))
    text_x_bearing, text_y_bearing, text_width, _, _, _ = context.text_extents(text_string)
    tx = text_x_bearing + text_width / 2.0
    ty = 0.3 * font_ascent
    x = - mea_size / 4.0 + inter_electrode_distance / 2.0 - tx
    y = + mea_size / 4.0 - inter_electrode_distance / 2.0 + ty
    context.move_to(x, y)
    context.set_source_rgb(0.0, 0.0, 0.0)  # i.e. black
    context.show_text(text_string)

    # Add horizontal line.
    # context.rectangle(-10.0, -10.0, 50.0, 50.0)
    # context.fill()
    # Store in file.
    surface.write_to_png("/tmp/test.png")
    # TODO add text.

    # Create white frame.
    shape = (height, width)
    frame = np.ones(shape)

    line_width = 10  # px

    # Add horizontal line.
    i_min = height // 2 - line_width // 2
    i_max = height // 2 + line_width // 2
    frame[i_min:i_max+1, :] = 0.0

    # Add vertical line.
    j_min = width // 2 - line_width // 2
    j_max = width // 2 + line_width // 2
    frame[:, j_min:j_max+1] = 0.0

    square_size = 500  # px
    line_width = 20  # px

    # Add square contour.
    i_top = height // 2 - square_size // 2
    i_bottom = height // 2 + square_size // 2
    j_left = width // 2 - square_size // 2
    j_right = width // 2 + square_size // 2
    # # Add top.
    i_min = i_top - line_width
    i_max = i_top - 0
    j_min = j_left - line_width
    j_max = j_right + line_width
    frame[i_min:i_max+1, j_min:j_max+1] = 0.0
    # # Add bottom.
    i_min = i_bottom + 0
    i_max = i_bottom + line_width
    j_min = j_left - line_width
    j_max = j_right + line_width
    frame[i_min:i_max+1, j_min:j_max+1] = 0.0
    # # Add left.
    i_min = i_top - line_width
    i_max = i_bottom + line_width
    j_min = j_left - line_width
    j_max = j_left - 0
    frame[i_min:i_max+1, j_min:j_max+1] = 0.0
    # # Add right.
    i_min = i_top - line_width
    i_max = i_bottom + line_width
    j_min = j_right + 0
    j_max = j_right + line_width
    frame[i_min:i_max+1, j_min:j_max+1] = 0.0

    # TODO remove the following lines.
    import matplotlib.pyplot as plt

    plt.imshow(frame, cmap='gray', vmin=0.0, vmax=1.0)
    print("plot")
    # plt.show()

    return
