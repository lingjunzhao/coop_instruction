This repos is adapted from Google Pangea (https://github.com/google-research/pangea) to collect human annotations from generated instructions.


### Prepare code and data

1. Clone this repo

2. Data preparation

In order to simulate an environment, PanGEA requires a dataset of 360-degree
panoramic images, plus navigation graphs encoding the position of these
viewpoints and the navigable connections between them. To use the Matterport3D
dataset, download the Matterport3D skybox images from
[here](https://niessner.github.io/Matterport/) and the navigation graphs from
[here](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
Please note that the Matterport3D data is governed by the following
[Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf). Alternatively,
a different dataset such as StreetLearn could be used by writing a new
dataloader class in `data_adapter.js`.

Save the data in directory as follows:

```bash
<data_dir>/data/v1/scans/<scan_id>/matterport_skybox_images/<pano_id>_skybox<face_id>_sami.jpg
<data_dir>/connectivity/<scan_id>_connectivity.json
```

create a symlink to the data, and creates upload directory for annotation outputs

```bash
ln -s <data_dir> symdata

mkdir <data_dir>/uploads
```


### Locally annotating paths from generated instructions

1. Specify input file

The input file is a json file, which the key is annotation id, and the value is the instruction to be annotated.

Edit `examples/follower_plugin.html` line 93:

```bash
const json_file = "../testdata/anno_speaker-clip-t5_greedy.json";
```

If the json key containing generated instruction is not `generated_instr`, make that consistent from line 94 too:

```bash
const instr_key = 'generated_instr'
```

2. Start server, open url and annotate 

Start a local server:

```bash
python3 -m http.server 8888
```

For example, if you want to annotate `id=1381` (corresponds to instruction `5057_0`) from `testdata/anno_speaker-clip-t5_greedy.json`,  browse to

```bash
http://localhost:8888/examples/follower_plugin.html?id=1381
```

3. Download annotation

After clicking `submit` button, the file would be saved to `uploads/${id}_snapshots_v1.json`.



