# galaxycnn
Term project for the course ASTR 502, Spring 2022, at the University of Arizona Steward Observatory.

**Instructors:** Prof Tim Eifler, Hung-Jin Huang

<!--- **Online documentation:** [https://400b-leach.readthedocs.io/en/latest/](https://400b-leach.readthedocs.io/en/latest/) --->


**Directory structure:**

- **`project`**: The research assignment. Top level is the report (LaTeX and PDF), code is in `project/notebooks`, graphics in `project/img`.
- **`zoobot`**: Essentially just the [code from Mike Walmsley]()
- **`data`:** Code to create suitable galaxy images for training, and a postgres database to keep track of them. The images themselves are too big to include in version control.
- **`doc`:** Sphinx-format files to make documentation (mostly automatically from the docstrings). Create locally with `make html` or `make latexpdf`, which put the resulting files in `doc/_build`. An online version is on ReadTheDocs, which may be more up to date.
<!---- **`source`:** The bulk of the Python code is in `source/galaxy`, as an installable module. --->