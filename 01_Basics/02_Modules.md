# MODULES on HPC

## Overview
The Lmod modules system on the HPC system enables users to easily set their environments for selected software and to choose versions if appropriate.

---

## Basic Commands

### List all available software
```bash
module avail
```

```
### Load the environment for a specific package
```bash
module load <package>
```
If no version is specified, the system loads the default version. To specify a version:
```bash
module load openmpi/4.1.4
```

### Remove a module
```bash
module unload <package>
```

### List all currently loaded modules
```bash
module list
```

### Switch from one module version to another
```bash
module swap <old_module> <new_module>
```
Example:
```bash
module swap openmpi/4.1.4 openmpi/5.0.6
```

### Clear all loaded modules
```bash
module purge
```

### Search for available versions of a module
```bash
module spider <package>
```
Example:
```bash
module spider namd/2.14
```

### Display detailed information about a module
```bash
module show <package>
```
Example:
```bash
module show mldl_modules/pytorch_gpu
```

### Display currently loaded environment variables
```bash
env
```

---

## Modules Best Practices
- **Start with a clean slate:** Use `module purge` before beginning a new workflow.
- **Specify exact versions:** Do not rely on defaults, as they may change.
- **Use `module spider` to check available versions** before loading a package.

---

## Advanced Usage

### Using a Bash Script for Modules
Instead of modifying `.bashrc`, use a script to load modules:
```bash
module purge
module load gcc/11.2.0
module load openmpi/3.1.4
```
Save it as `modules.sh` and run:
```bash
source modules.sh
```

For more details, refer to the HPC documentation (https://lmod.readthedocs.io/en/latest/020_advanced.html).

