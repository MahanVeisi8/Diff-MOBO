# Docker + OpenFOAM Setup Guide

This guide explains how to set up and run OpenFOAM inside Docker for the Generative Airfoil project.

---

## 0. Initialize Docker

### a. Download the image
Download the Docker image file (`airfoil_docker.tar`) from [this link](https://nextcloud.mpi-klsb.mpg.de/index.php/s/GCiYgqGaQYjcwkE?path=%2F).

### b. Go to the project directory
From the repository you cloned:

```bash
cd src/OpenFoam
````

### c. Copy the image

Place `airfoil_docker.tar` into the `src/OpenFoam/` directory.

### d. Load the image

```bash
docker image load -i airfoil_docker.tar
```

### e. Run a container

Start a container that mounts the repository:

```bash
docker run -d -it \
  --name airfoil_mount \
  -v "$(pwd):/home/airfoil_UANA" \
  airfoil_mount_2
```

* Replace `$(pwd)` with the absolute path of your project on Windows (`${PWD}` in PowerShell).
* `--name airfoil_mount` gives the container a friendly name.

**Explanation:** Mounts the directory with your data and Python scripts into `/home/airfoil_UANA` inside the container.

---

## 1. Explore the container

Check running containers:

```bash
docker ps -a
```

Open a bash shell inside:

```bash
docker exec -it airfoil_mount /bin/bash
```

---

## 2. Inside the container

Run these once:

```bash
cd /home/airfoil_UANA
source /opt/openfoam5/etc/bashrc

pip3 install --upgrade pip
pip3 install tensorflow==1.15   # only if required
pip3 install pymoo
```

Exit the container:

```bash
exit
```

---

## 3. Simulation setup

Navigate to `src/OpenFoam/Airfoil_simulation_1/`.

* You will find a directory named `OpenFOAM_0`.
* This directory manages one CPU core for parallel runs.
* If you have `N` cores (e.g., 200), copy and rename the folder `OpenFOAM_j` for each core `j`.

> This enables near-linear scaling (\~200× faster if using 200 cores).

---

## 4. Python dependencies (host)

From the project root:

```bash
pip install -r src/optimization_loop/requirements.txt
```

---

## 5. Run the outer loop

Update the container name in `src/optimization_loop/outerloop_creation.py` if needed.

Then run:

```bash
python src/optimization_loop/outerloop_creation.py
```

Results will be saved to:

```
src/optimization_loop/run_results.npy
```

---

## Troubleshooting

* **Permission denied for `Allclean`:**

  ```bash
  chmod +x Allclean
  ```

* **Kill stray processes (use with care):**

  ```bash
  pkill -u root
  ```

* **Restart a stopped container:**

  ```bash
  docker start <container_id>
  ```

---

## Notes

* Outer loop can also run in **surrogate mode** for quick testing (see project root README).
* For any questions, contact **Bardia** (always on-call).

```




