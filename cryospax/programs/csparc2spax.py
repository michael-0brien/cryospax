"""Convert a CryoSPARC `.cs` file into a STAR file readable by cryospax."""

import argparse
import pathlib
import sys

from cryospax import read_csparc_file_as_starfile, write_starfile


DESCRIPTION = """\
Convert a CryoSPARC '.cs' file into a RELION-style STAR file that can be read
with a `cryospax.RelionParticleParameterFile`.

This is *not* a general purpose '.cs'-to-STAR converter. Only the CryoSPARC
fields that cryospax knows how to interpret are written out, i.e. those needed
to build the image configuration, the CTF, and the pose of each particle. Every
other field of the '.cs' file is dropped, so the result is not a substitute for
the original file and is not guaranteed to be understood by RELION or by any
other program that expects a complete STAR file.
"""

EPILOG = """\
example:
  cryospax csparc2spax P1/J42/particles.cs particles.star \\
      --passthrough P1/J42/passthrough_particles.cs
"""


def main():
    parser = argparse.ArgumentParser(
        prog="cryospax csparc2spax",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csfile",
        type=pathlib.Path,
        help="path to the CryoSPARC '.cs' file to convert",
    )
    parser.add_argument(
        "starfile",
        type=pathlib.Path,
        help="path where to write the output STAR file, including a '.star' suffix",
    )
    parser.add_argument(
        "--passthrough",
        type=pathlib.Path,
        default=None,
        metavar="CSFILE",
        help=(
            "path to an optional CryoSPARC passthrough particles '.cs' file, whose "
            "fields are merged into those of `csfile` on the particle 'uid'"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the output STAR file if it already exists",
    )
    args = parser.parse_args()

    if args.starfile.exists() and not args.overwrite:
        parser.error(
            f"STAR file '{args.starfile}' already exists. Pass `--overwrite` to erase it."
        )

    try:
        starfile_data = read_csparc_file_as_starfile(args.csfile, args.passthrough)
        write_starfile(starfile_data, args.starfile)
    except (OSError, ValueError) as error:
        # Reading a `.cs` file that cryospax cannot convert is a user error, not a
        # bug, so report it without a traceback
        parser.exit(status=1, message=f"error: {error}\n")

    num_particles = len(starfile_data["particles"])
    num_optics_groups = len(starfile_data["optics"])
    print(
        f"Wrote {num_particles} particles in {num_optics_groups} optics group(s) to "
        f"'{args.starfile}'."
    )


if __name__ == "__main__":
    sys.exit(main())
