import numpy as np
import os

header_fields = (
    ('filename', 'S1', 20),
    ('parent_filename', 'S1', 20),
    ('comments1', 'S1', 80),
    ('comments2', 'S1', 80),
    ('energy_type', np.int16, 1),
    ('config_type', np.int16, 1),
    ('file_type', np.int16, 1),
    ('trans_type', np.int16, 1),
    ('scan_type', np.int16, 1),
    ('data_type', np.int16, 1),
    ('date_modified', 'S1', 16),
    ('frequency', np.float32, 1),
    ('mat_velocity', np.float32, 1),
    ('num_pts', np.int32, 1),
    ('num_polarization_channels', np.int16, 1),
    ('spare00', np.int16, 1),
    ('adc_min_voltage', np.float32, 1),
    ('adc_max_voltage', np.float32, 1),
    ('band_width', np.float32, 1),
    ('spare01', np.int16, 5),
    ('polarization_type', np.int16, 4),
    ('record_header_size', np.int16, 1),
    ('word_type', np.int16, 1),
    ('word_precision', np.int16, 1),
    ('min_data_value', np.float32, 1),
    ('max_data_value', np.float32, 1),
    ('avg_data_value', np.float32, 1),
    ('data_scale_factor', np.float32, 1),
    ('data_units', np.int16, 1),
    ('surf_removal', np.uint16, 1),
    ('edge_weighting', np.uint16, 1),
    ('x_units', np.uint16, 1),
    ('y_units', np.uint16, 1),
    ('z_units', np.uint16, 1),
    ('t_units', np.uint16, 1),
    ('spare02', np.int16, 1),
    ('x_return_speed', np.float32, 1),
    ('y_return_speed', np.float32, 1),
    ('z_return_speed', np.float32, 1),
    ('scan_orientation', np.int16, 1),
    ('scan_direction', np.int16, 1),
    ('data_storage_order', np.int16, 1),
    ('scanner_type', np.int16, 1),
    ('x_inc', np.float32, 1),
    ('y_inc', np.float32, 1),
    ('z_inc', np.float32, 1),
    ('t_inc', np.float32, 1),
    ('num_x_pts', np.int32, 1),
    ('num_y_pts', np.int32, 1),
    ('num_z_pts', np.int32, 1),
    ('num_t_pts', np.int32, 1),
    ('x_speed', np.float32, 1),
    ('y_speed', np.float32, 1),
    ('z_speed', np.float32, 1),
    ('x_acc', np.float32, 1),
    ('y_acc', np.float32, 1),
    ('z_acc', np.float32, 1),
    ('x_motor_res', np.float32, 1),
    ('y_motor_res', np.float32, 1),
    ('z_motor_res', np.float32, 1),
    ('x_encoder_res', np.float32, 1),
    ('y_encoder_res', np.float32, 1),
    ('z_encoder_res', np.float32, 1),
    ('date_processed', 'S1', 8),
    ('time_processed', 'S1', 8),
    ('depth_recon', np.float32, 1),
    ('x_max_travel', np.float32, 1),
    ('y_max_travel', np.float32, 1),
    ('elevation_offset_angle', np.float32, 1),
    ('roll_offset_angle', np.float32, 1),
    ('z_max_travel', np.float32, 1),
    ('azimuth_offset_angle', np.float32, 1),
    ('adc_type', np.int16, 1),
    ('spare06', np.int16, 1),
    ('scanner_radius', np.float32, 1),
    ('x_offset', np.float32, 1),
    ('y_offset', np.float32, 1),
    ('z_offset', np.float32, 1),
    ('t_delay', np.float32, 1),
    ('range_gate_start', np.float32, 1),
    ('range_gate_end', np.float32, 1),
    ('ahis_software_version', np.float32, 1),
    ('spare_end', np.float32, 5)
)

def get_header(f):
    header = dict()
    for field, dtype, count in header_fields:
        data = np.fromfile(f, dtype=dtype, count=count)
        header[field] = b''.join(data) if dtype is 'S1' else data

    return header

def read_a3d(fname):
    assert os.path.splitext(fname)[1] == '.a3d', \
            "{} does not have a '.a3d' extension".format(fname)
    with open(fname, 'r+b') as f:
        h = get_header(f)
        nx = int(h['num_x_pts'])
        ny = int(h['num_y_pts'])
        nt = int(h['num_t_pts'])
        if(h['word_type']==7):      #float32
            data = np.fromfile(f, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4):    #uint16
            data = np.fromfile(f, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F') #make N-d image
        return data
