# galaxycnn
Term project for the course ASTR 502, Spring 2022, at the University of Arizona Steward Observatory.

**Instructors:** Prof Tim Eifler, Hung-Jin Huang

**Directory structure:**

- **`project_report`**: Report (LaTex) and slides (Powerpoint), with graphics in `project/figures`.
- **`zoobot`**: Essentially just the [code from Mike Walmsley]()
- **`projcode`**: Code specifically for ASTR 502.
- **`notebooks`**: Jupyter notebooks as practice for transferring to Colab.

**Installation**:

To maintain the namespace already used within zoobot, the two packages need to be installed separately. From the proj502 top directory:
```bash
pip install -r requirements.txt
pip install -e zoobot
pip install -e projcode
```

This adds the `zoobot` and `projcode` subdirectories to `sys.path`, but on my system that wasn't enough to avoid ModuleNotFoundError problems (reasons unclear). One fix is to add the project root to PYTHONPATH, for example 

`export PYTHONPATH="$PYTHONPATH:$HOME/path/to/proj502/"`

Either put this in `.bashrc` (or your OS's equivalent), or run it from the command line before any python code.
