data_mount_and_image_bucketing = {
        'data_path': './data/', 
        'data_mount': {
            'password': '****',
            'mount_point': '/mnt/usb',
            'mounted_drive': '/dev/sda4',
            'data_path': '/mnt/usb/python/project_dataset/utkface_aligned_cropped/UTKFace/'
            },

        'image_bucketing': {
            'no_of_buckets': 5,
            'batch_size': 5_000
            }

        }
