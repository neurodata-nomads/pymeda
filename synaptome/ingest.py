import numpy as np
import intern
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource


def ingest_volume(host, token, channel_name, collection, experiment, volume):
    """
        Assumes the collection and experiment exists in BOSS.
        """

    remote = BossRemote({'protocol': 'https', 'host': host, 'token': token})

    if volume.dtype == 'uint64':
        dtype = 'uint64'
        img_type = 'annotation'
        sources = ['empty']
    else:
        dtype = volume.dtype.name
        img_type = 'image'
        sources = []

    try:
        channel_resource = ChannelResource(channel_name, collection,
                                           experiment)
        channel = remote.get_project(channel_resource)
    except:
        channel_resource = ChannelResource(
            channel_name,
            collection,
            experiment,
            type=img_type,
            sources=sources,
            datatype=dtype)
        channel = remote.create_project(channel_resource)

    #Get max size of experiment
    exp_resource = ExperimentResource(experiment, collection)
    coord_frame = remote.get_project(exp_resource).coord_frame
    coord_frame_resource = CoordinateFrameResource(coord_frame)
    data = remote.get_project(coord_frame_resource)
    y_stop, x_stop = data.y_stop, data.x_stop

    for z in range(volume.shape[0]):
        print('Uploading {} slice'.format(z))
        remote.create_cutout(channel, 0, (0, x_stop), (0, y_stop), (z, z + 1),
                             volume[z, :, :].reshape((-1, y_stop, x_stop)))
