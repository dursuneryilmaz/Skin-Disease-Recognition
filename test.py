import pre_processing
import model_train
import live_cam

src_dir = 'HAM10000'
dst_dir = 'image_data'

# pre_processing.split_to_dirs(src_dir, dst_dir)
# pre_processing.etl_for_all(dst_dir)

# model_train.create_model()

live_cam.start()
