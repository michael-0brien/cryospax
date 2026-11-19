# The command line interface

CryoSPAX programs are run through the `cryospax` command

```bash
cryospax program-name ...
```

Pass `--help` to any program to see its arguments, e.g. `cryospax csparc2spax --help`.

## `csparc2spax`

Convert a CryoSPARC `.cs` file into a RELION STAR file that can be read with a
[`cryospax.RelionParticleParameterFile`][].

```bash
cryospax csparc2spax particles.cs particles.star --passthrough passthrough_particles.cs
```

**Arguments:**

- `csfile`:
    The path to the CryoSPARC `.cs` file to convert. This must include a '.cs'
    extension.
- `starfile`:
    The path where to write the STAR file. This must include a '.star' extension.
- `--passthrough`:
    An optional CryoSPARC passthrough particles `.cs` file, whose fields are
    merged into those of `csfile` on the particle `'uid'`. Fields present in both
    files are taken from `csfile`.
- `--overwrite`:
    Replace the output file if it already exists. Without this, an existing STAR
    file is never overwritten.

!!! warning "This is not a general purpose converter"

    Only the CryoSPARC fields that CryoSPAX interprets are written out, i.e. those
    needed to build the image configuration, the CTF, and the pose of each
    particle. Every other field is dropped, so the result is not a substitute for
    the original file and is not guaranteed to be understood by RELION or by other
    programs that expect a complete STAR file.

    For a general purpose conversion, use
    [`pyem`](https://github.com/asarnow/pyem), after which this program is named.

### Reading the result

```python
from cryospax import RelionParticleDataset, RelionParticleParameterFile

parameter_file = RelionParticleParameterFile("particles.star")
dataset = RelionParticleDataset(parameter_file, "/path/to/cryosparc/project")
```

Image paths are taken from the CryoSPARC `'blob/path'` field, so the project
directory passed to the dataset is the *CryoSPARC* project directory.
