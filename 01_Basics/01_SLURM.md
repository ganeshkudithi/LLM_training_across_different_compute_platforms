# SLURM: A Guide to Job Scheduling on HPC

## Introduction

### What is Slurm?
Slurm(Simple Linux Utility for Resource Management), is a powerful computational workload scheduler used on many of the world's largest supercomputers. It manages computing resources efficiently, ensuring fair access to available nodes in a cluster. 

#### Think of it like:
- **A Laboratory Resource Manager**: Similar to how a lab manages shared equipment like microscopes and sequencers, Slurm controls shared computer resources in an HPC cluster.

For the latest documentation, visit the [official Slurm website](https://slurm.schedmd.com/documentation.html).

---

## Terminology

- **Job**: A unit of work submitted to Slurm, consisting of one or more tasks.
- **Task**: A single executable program within a job.
- **Partition**: A logical division of the cluster, each with its own resource limits and policies.
- **Reservation**: A block of computing resources pre-allocated for specific jobs or users.
- **Node**: A single computer within a cluster containing CPU, RAM, and possibly a GPU.
- **Compute Node**: Nodes that execute jobs, each containing multiple **sockets**.
- **Socket**: A physical slot in a node where a CPU is installed.
- **Core**: A single physical processor unit within a CPU.
- **Thread**: A sequence of instructions processed independently by a core.

---

## Key Functions of Slurm

### Job Handling:
- Allows users to submit jobs with specific resource requirements.
- Evaluates, prioritizes, and schedules jobs.
- Provides tools to monitor and manage running jobs.

### Resource Management:
- Allocates cluster resources (CPUs, memory, GPUs) to jobs.
- Manages partitions.
- Tracks resource usage for reporting and accounting.

### Typical Workflow:
1. **User Actions:**
   - Log into the HPC system.
   - Create a job script defining resources.
   - Submit the job to Slurm.
   - Monitor the job.
2. **Slurm Actions:**
   - Receives and queues the job.
   - Matches requirements to available resources.
   - Schedules and executes the job.
   - Tracks job status and resource usage.


## Slurm Job States

Jobs go through several states while being processed. Below are common states:

- **PENDING (PD):** Job is waiting in the queue for resources.
- **CONFIGURING (CF):** Job is being set up before execution.
- **RUNNING (R):** Job is actively executing.
- **SUSPENDED (S):** Job is temporarily paused but retains resources.
- **COMPLETING (CG):** Job is finalizing before ending.
- **COMPLETED (CD):** Job finished successfully.
- **CANCELLED (CA):** Job was manually cancelled.
- **FAILED (F):** Job terminated with an error.
- **TIMEOUT (TO):** Job exceeded the allocated time.

For more job states, visit [Slurm's documentation](https://slurm.schedmd.com/squeue.html).

---

## Basic Slurm Commands

### Job Submission and Management
```bash
sbatch <job_script>
``` 
Submit a batch job.

```bash
srun <command>
``` 
Run a command interactively on allocated resources.

```bash
squeue -u <username>
``` 
Check the status of your jobs.

```bash
scancel <job_id>
``` 
Cancel a running job.

```bash
scontrol show job <job_id>
``` 
Show detailed job information.

### Checking Cluster Status
```bash
sinfo
``` 
View available nodes and partitions.

```bash
squeue
``` 
View all running and pending jobs.


### Resource Management
```bash
module load <package>
``` 
Load a specific software package.

```bash
module list
``` 
List currently loaded modules.

```bash
module purge
``` 
Unload all loaded modules.

---

## Writing Slurm Job Scripts

### Basic Sample Script
```bash
#!/bin/bash
#SBATCH --job-name=serialJob
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --time=0-00:30:00
#SBATCH --output=serialJob.%j.out
#SBATCH --error=serialJob.%j.err

module load miniconda3
python my_script.py
```


### GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=gpu-job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=gpu-job.%j.out

module load miniconda3
python my_script.py
```

---

## Troubleshooting

### Common Issues and Fixes

1. **Job Stuck in PENDING (PD):**
   - Use `squeue` to check reasons (e.g., waiting for resources).
   - Check if another job is blocking yours.

2. **Job Crashes Unexpectedly:**
   - Check output/error logs (`.out` and `.err` files).
   - Ensure all required modules and software are loaded.

3. **Application-Specific Problems:**
   - Debug your script locally before submitting.
   - Use interactive jobs (`srun --pty /bin/bash`) for debugging.

---

## Additional Resources

### Official Documentation
- [Slurm User Guide](https://slurm.schedmd.com/quickstart.html)
- [Slurm Commands Summary](https://slurm.schedmd.com/pdfs/summary.pdf)
